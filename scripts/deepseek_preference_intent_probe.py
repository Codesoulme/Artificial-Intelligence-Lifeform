"""Calibrate a restricted current-preference read gate over frozen features."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-preference-intent-20261110'
FORMAT = 'deepseek-finite-preference-intent-development-v1'

TRAIN_NAMES = NAMES[:6]
CAL_NAMES = ('阿松', '小雪', '阿禾', '小青')
TEST_NAMES = ('林栩', '许澄', '江屿', '叶芮')
TRAIN_POSITIVE = ('{name}喜欢喝什么？', '{name}最喜欢的饮料是什么？', '请说出{name}爱喝的饮品。',
                  '{name}现在最爱喝什么？', '告诉我{name}喜欢的饮料。', '{name}偏好的饮料是啥？')
TRAIN_NEGATIVE = {
    'negated': ('不要告诉我{name}喜欢什么饮料。', '我不想知道{name}喜欢喝什么。'),
    'history': ('{name}以前喜欢喝什么？', '查询{name}过去最爱的饮料。'),
    'source': ('谁告诉你{name}喜欢什么饮品？',),
    'write': ('记住{name}喜欢喝水。', '请修改{name}喜欢的饮料。'),
    'multiple': ('{name}和阿城各自喜欢什么饮料？',),
    'derived': ('{name}喜欢喝水吗？', '{name}最喜欢的饮料是茶吗？', '{name}偏爱的饮品是不是果汁？'),
    'missing': ('喜欢喝什么？', '现在最爱的饮品是什么？', '目前最喜欢什么饮料？'),
}
CAL_POSITIVE = ('{name}偏爱的饮品是什么？', '我想知道{name}当前爱喝什么。')
CAL_NEGATIVE = {
    'negated': ('暂时别说{name}爱喝什么。',),
    'history': ('{name}从前最爱什么饮料？',),
    'source': ('{name}饮品喜好的消息来自哪里？',),
    'write': ('将{name}的饮料喜好改成茶。',),
    'multiple': ('比较{name}和阿城最爱的饮料。',),
    'derived': ('{name}爱喝的是咖啡吗？',),
}
TEST_POSITIVE = ('{name}最中意哪种饮料？', '请告诉我{name}现在最喜欢喝什么。',
                 '说说{name}偏好喝的东西。')
TEST_NEGATIVE = {
    'negated': ('请不要回答{name}最爱喝什么。',),
    'history': ('{name}原先的饮品喜好是什么？',),
    'source': ('关于{name}饮料偏好的记录是谁提供的？',),
    'write': ('把{name}喜欢的饮品更新为果汁。',),
    'multiple': ('{name}与阿城分别爱喝什么？',),
    'derived': ('{name}最爱的饮料是不是水？',),
    'conditional': ('如果{name}喜欢茶，就说一声。',),
    'chat': ('你好，请给我一个放松建议。',),
    'missing': ('现在最喜欢喝什么？',),
}


def cases(names, positive, negative):
    rows = [{'text': template.format(name=name), 'label': 1, 'group': 'current'}
            for name in names for template in positive]
    for group, templates in negative.items():
        for template in templates:
            if '{name}' in template:
                rows.extend({'text': template.format(name=name), 'label': 0, 'group': group} for name in names)
            else:
                rows.append({'text': template, 'label': 0, 'group': group})
    return rows


def calibrate_threshold(scores, labels):
    if scores.ndim != 1 or labels.shape != scores.shape or not bool(torch.isfinite(scores).all()):
        raise ValueError('Expected finite scores and matching labels')
    negatives = scores[labels == 0]
    if not len(negatives) or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Calibration requires binary labels and negatives')
    threshold = torch.nextafter(negatives.max(), torch.tensor(float('inf'), dtype=scores.dtype, device=scores.device))
    if not bool(torch.isfinite(threshold)):
        raise ValueError('Cannot construct a finite threshold')
    return float(threshold)


def evaluate(scores, rows, threshold):
    if scores.ndim != 1 or len(scores) != len(rows) or not bool(torch.isfinite(scores).all()):
        raise ValueError('Expected finite score per case')
    result = [dict(row, logit=float(score), accepted=float(score) >= threshold)
              for row, score in zip(rows, scores)]
    summary = {}
    for group in sorted({r['group'] for r in result}):
        selected = [r for r in result if r['group'] == group]
        summary[group] = {'examples': len(selected), 'accepted': sum(r['accepted'] for r in selected),
                          'correct': sum(r['accepted'] == bool(r['label']) for r in selected)}
    return {'summary': summary, 'rows': result}


def run(output=OUTPUT):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        raise ValueError('Expected a new output directory under runs')
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    config_sha = feature_files_sha256(MODEL)
    splits = {'train': cases(TRAIN_NAMES, TRAIN_POSITIVE, TRAIN_NEGATIVE),
              'calibration': cases(CAL_NAMES, CAL_POSITIVE, CAL_NEGATIVE),
              'test': cases(TEST_NAMES, TEST_POSITIVE, TEST_NEGATIVE)}
    if any({r['text'] for r in splits[a]} & {r['text'] for r in splits[b]}
           for a, b in (('train', 'calibration'), ('train', 'test'), ('calibration', 'test'))):
        raise ValueError('Text overlap between splits')
    torch.manual_seed(4642)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for name, rows in splits.items():
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        vectors[name] = pool(features, masks)
        print(f'{name} features ready: {len(rows)}', flush=True)
    gate = nn.Linear(vectors['train'].shape[-1], 1)
    labels = torch.tensor([r['label'] for r in splits['train']], dtype=torch.float32)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(gate(vectors['train']).squeeze(-1), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
    gate.eval()
    with torch.no_grad():
        logits = {name: gate(vectors[name]).squeeze(-1) for name in splits}
        threshold = calibrate_threshold(logits['calibration'], torch.tensor([r['label'] for r in splits['calibration']]))
        results = {name: evaluate(logits[name], rows, threshold) for name, rows in splits.items()}
    if (any(sha(p) != digest for p, digest in protected.items()) or frozen_digest(organism, original) != before
            or feature_files_sha256(MODEL) != config_sha or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    checkpoint = output / 'gate.pt'
    torch.save({'format': FORMAT, 'state': gate.state_dict(), 'hidden_width': gate.in_features,
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18], 'threshold_logit': threshold}, checkpoint)
    report = {'status': 'restricted_intent_development_only', 'gate_sha256': sha(checkpoint),
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'feature_config_sha256': config_sha, 'threshold_logit': threshold, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Handwritten fixed-relation samples, not open chat or arbitrary facts.',
                              'Threshold is calibrated to reject the finite negative calibration set only.',
                              'No entity binding, state status or actual generation evaluated here.',
                              'Model pretraining overlap and shared semantic templates are unknown.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: result['summary'] for name, result in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
