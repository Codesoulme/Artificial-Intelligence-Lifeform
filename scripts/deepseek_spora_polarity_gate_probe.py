"""Develop a polarity/claim-type gate for composed SPORA memory parsing."""
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

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_value_span_probe import feature_batch
from scripts.deepseek_spora_value_span_probe import UNKNOWN
from scripts.deepseek_spora_observation_retrain import SPLITS
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-polarity-gate-20261111'
CLASSES = ('affirmative', 'negation', 'hypothesis', 'third_party', 'action', 'question_history')
NEGATIVE = {
    'train': {
        'negation': ('{n}不喜欢喝{w}。', '{n}从来不喝{w}。', '{n}以后不再喜欢{w}。'),
        'hypothesis': ('我猜{n}可能喜欢{w}。', '假如{n}喜欢{w}就好了。'),
        'third_party': ('{n}说朋友喜欢{w}。',),
        'action': ('{n}给朋友买了{w}。', '{n}为客人准备了{w}。'),
        'question_history': ('{n}是否爱喝{w}？', '{n}以前喝{w}，现在不清楚。'),
    },
    'calibration': {
        'negation': ('{n}完全不喝{w}。',), 'hypothesis': ('也许{n}喜欢{w}。',),
        'third_party': ('听说{n}喜欢{w}。',), 'action': ('{n}替同事带了{w}。',),
        'question_history': ('你知道{n}是否爱喝{w}吗？',),
    },
    'held': {
        'negation': ('{n}从来不喝{w}。', '{n}不再喜欢{w}了。'),
        'hypothesis': ('如果{n}喜欢{w}就好了。',), 'third_party': ('{n}说阿明喜欢{w}。',),
        'action': ('{n}给客人准备了{w}。',), 'question_history': ('{n}曾经喝{w}，后来不清楚。',),
    },
}


def build(split):
    names, positives, _ = SPLITS[split]
    words = ('茶', '咖啡', '牛奶', '果汁', '豆浆', '可乐', '水', '酸奶') + UNKNOWN[split]
    rows = []
    for name in names:
        for word in words:
            for template in positives:
                rows.append({'text': template.format(n=name, w=word), 'label': 0, 'kind': 'affirmative'})
            for kind, templates in NEGATIVE[split].items():
                for template in templates:
                    rows.append({'text': template.format(n=name, w=word), 'label': CLASSES.index(kind), 'kind': kind})
    return rows


def pool(vectors, mask):
    weights = mask[:, None, :, None].float()
    means = (vectors.float() * weights).sum(2) / weights.sum(2).clamp_min(1)
    return torch.cat([F.layer_norm(means[:, i], (means.shape[-1],)) for i in range(2)], dim=-1)


def evaluate(model, vectors, rows, threshold):
    with torch.inference_mode():
        probabilities = model(vectors).softmax(-1)
        confidence, prediction = probabilities.max(-1)
        accepted = (prediction == 0) & (probabilities[:, 0] >= threshold)
    result = [{**row, 'prediction': int(prediction[i]), 'confidence': float(confidence[i]),
               'affirmative_probability': float(probabilities[i, 0]), 'accepted': bool(accepted[i])}
              for i, row in enumerate(rows)]
    return {'positive_total': sum(row['label'] == 0 for row in result),
            'positive_accepted': sum(row['label'] == 0 and row['accepted'] for row in result),
            'negative_total': sum(row['label'] != 0 for row in result),
            'negative_accepted': sum(row['label'] != 0 and row['accepted'] for row in result),
            'rows': result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    datasets = {split: build(split) for split in ('train', 'calibration', 'held')}
    paths = (MODEL / 'model.safetensors', CHECKPOINT)
    source = {str(path): sha(path) for path in paths}
    if source[str(paths[0])] != WEIGHT_SHA:
        raise ValueError('Base provenance mismatch')
    torch.manual_seed(90115)
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, organism, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    vectors = {}
    for split, rows in datasets.items():
        features, masks, _, _ = feature_batch(base, tokenizer, rows)
        vectors[split] = pool(features, masks)
        print(split, len(rows), flush=True)
    model = nn.Linear(vectors['train'].shape[-1], len(CLASSES))
    labels = torch.tensor([row['label'] for row in datasets['train']])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(vectors['train']), labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = model(vectors['calibration']).softmax(-1)[:, 0]
    thresholds = sorted(set(float(value) for value in probabilities))
    thresholds.append(thresholds[-1] + 0.01)
    candidates = [(evaluate(model, vectors['calibration'], datasets['calibration'], threshold), threshold)
                  for threshold in thresholds]
    feasible = [(result, threshold) for result, threshold in candidates if result['negative_accepted'] == 0]
    selected, threshold = max(feasible, key=lambda pair: (pair[0]['positive_accepted'], -pair[1]))
    results = {split: evaluate(model, vectors[split], rows, threshold) for split, rows in datasets.items()}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected input changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'gate.pt'
    torch.save({'format': 'spora-claim-polarity-gate-development-v1', 'state': model.state_dict(),
                'threshold': threshold, 'classes': CLASSES, 'base_sha256': WEIGHT_SHA,
                'source_checkpoint_sha256': source[str(CHECKPOINT)], 'feature_variant_sha256': variant}, checkpoint)
    report = {'status': 'research_only_no_writes', 'threshold': threshold, 'classes': CLASSES,
              'checkpoint_sha256': sha(checkpoint), 'source_hashes': source,
              'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks),
              'results': results, 'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic templates and finite names/values.',
                              'No independent dialogue, source trust, temporal resolution or write authorization.',
                              'Threshold selected on finite calibration data and is not a safety guarantee.',
                              'No persistent memory writes.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, result in results.items():
        print(split, {key: result[key] for key in ('positive_accepted', 'positive_total', 'negative_accepted', 'negative_total')}, flush=True)


if __name__ == '__main__':
    main()
