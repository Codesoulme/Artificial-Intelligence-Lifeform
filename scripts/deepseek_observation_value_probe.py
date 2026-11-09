"""Research probe for bounded Chinese observations and abstention."""
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
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256

OUTPUT = ROOT / 'runs/deepseek-observation-value-probe-20261110'
ABSTAIN = len(WORDS)
TRAIN_NAMES = ('小林', '阿宁', '小雨', '阿哲')
HELD_NAMES = ('林栩', '许澄')
TRAIN_POSITIVE = ('{name}喜欢喝{word}。', '{name}最喜欢的饮料是{word}。', '我知道{name}爱喝{word}。')
TRAIN_NEGATIVE = {
    'negation': ('{name}不喜欢喝{word}。',),
    'question': ('{name}喜欢喝{word}吗？',),
    'hypothesis': ('假如{name}喜欢喝{word}就好了。',),
}
EXTRA_POSITIVE = ('{name}最爱饮用{word}。', '{name}常说自己爱喝{word}。',
                  '{name}最中意的饮品是{word}。', '记住，{name}喜欢的饮料是{word}。',
                  '关于{name}：最喜欢喝{word}。')
EXTRA_NEGATIVE = {
    'negation': ('{name}喜欢的饮料绝不是{word}。',),
    'history': ('{name}曾喜欢{word}，现在的喜好未知。',),
    'uncertain': ('我不确定{name}是不是喜欢{word}。', '我推测{name}可能喜欢喝{word}。',
                  '有人猜{name}也许喜欢{word}，没有证实。', '我猜{name}大概爱喝{word}，但不确定。'),
}
HELD_POSITIVE = ('{name}偏爱的饮品是{word}。', '据我所知，{name}最爱喝{word}。')
HELD_NEGATIVE = {
    'negation': ('{name}一点也不爱喝{word}。',),
    'question': ('你知道{name}是否爱喝{word}吗？',),
    'hypothesis': ('也许{name}爱喝{word}，我不确定。',),
    'history': ('{name}以前喜欢喝{word}，现在不知道了。',),
    'other_person': ('不是{name}，而是小周喜欢喝{word}。',),
    'multiple': ('{name}喜欢喝{word}还是牛奶，我忘记了。',),
}


def cases(names, positives, negatives):
    rows = []
    for name in names:
        for value, word in enumerate(WORDS):
            rows.extend({'text': template.format(name=name, word=word), 'label': value, 'mentioned_value': value, 'group': 'positive'}
                        for template in positives)
            for group, templates in negatives.items():
                rows.extend({'text': template.format(name=name, word=word), 'label': ABSTAIN, 'mentioned_value': value, 'group': group}
                            for template in templates)
    return rows


def pool(features, masks):
    mask = masks[:, None, :, None].float()
    means = (features.float() * mask).sum(2) / mask.sum(2).clamp_min(1)
    return torch.cat([F.layer_norm(means[:, i], (means.shape[-1],)) for i in range(2)], dim=-1)


def metrics(logits, rows):
    predicted = logits.argmax(-1).tolist()
    result = []
    for row, prediction in zip(rows, predicted):
        result.append({**row, 'predicted': prediction, 'correct': prediction == row['label']})
    summary = {}
    for group in sorted({r['group'] for r in result}):
        selected = [r for r in result if r['group'] == group]
        summary[group] = {'examples': len(selected), 'correct': sum(r['correct'] for r in selected),
                          'would_write': sum(r['predicted'] != ABSTAIN for r in selected)}
    return {'summary': summary, 'rows': result}


def run(output=OUTPUT, augmented=False, diverse=False, abstain_weight=1.0):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        raise ValueError('Expected a new output directory under runs')
    if diverse and not augmented or not 0.05 <= abstain_weight <= 2.0:
        raise ValueError('Diverse training requires augmentation and a valid abstain weight')
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    config_sha = feature_files_sha256(MODEL)
    train, held = cases(TRAIN_NAMES, TRAIN_POSITIVE, TRAIN_NEGATIVE), cases(HELD_NAMES, HELD_POSITIVE, HELD_NEGATIVE)
    if diverse:
        train += cases(TRAIN_NAMES, EXTRA_POSITIVE, EXTRA_NEGATIVE)
    if augmented:
        train += [{'text': template.format(name=name, word=word), 'label': ABSTAIN, 'group': 'unsupported'}
                  for name in TRAIN_NAMES for word in ('茉莉花茶', '苏打水', '苹果汁', '冰牛奶', '绿豆汤')
                  for template in TRAIN_POSITIVE + (EXTRA_POSITIVE if diverse else ())]
    held += [{'text': f'{name}喜欢喝{word}。', 'label': ABSTAIN, 'group': 'unsupported'}
             for name in HELD_NAMES for word in ('柠檬水', '热巧克力', '汽水', '白开水')]
    if set(r['text'] for r in train) & set(r['text'] for r in held):
        raise ValueError('Training and held texts overlap')
    torch.manual_seed(4640)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    vectors = {}
    for name, rows in (('train', train), ('held', held)):
        features, masks = extract(base, tokenizer, [row['text'] for row in rows], batch_size=2)
        vectors[name] = pool(features, masks)
        print(f'{name} features ready: {len(rows)}', flush=True)
    classifier = nn.Linear(vectors['train'].shape[-1], ABSTAIN + 1)
    labels = torch.tensor([row['label'] for row in train])
    weights = torch.ones(ABSTAIN + 1)
    weights[ABSTAIN] = abstain_weight
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.0002, weight_decay=0.01)
    for step in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(vectors['train']), labels, weight=weights)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.no_grad():
        results = {name: metrics(classifier(vectors[name]), rows) for name, rows in (('train', train), ('held', held))}
    if (any(sha(p) != digest for p, digest in protected.items()) or frozen_digest(organism, original) != before
            or feature_files_sha256(MODEL) != config_sha or any(p.grad is not None for p in base.parameters())):
        raise ValueError('Protected input changed')
    output.mkdir(parents=False, exist_ok=False)
    checkpoint = output / 'classifier.pt'
    torch.save({'format': 'deepseek-finite-observation-value-development-v1',
                'state': classifier.state_dict(), 'vocabulary': WORDS, 'abstain': ABSTAIN,
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'feature_config_sha256': config_sha, 'hidden_indices': [6, 18]}, checkpoint)
    report = {'status': 'research_only_no_writes', 'classifier_sha256': sha(checkpoint),
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'feature_config_sha256': config_sha, 'steps': 600, 'augmented': augmented, 'diverse': diverse,
              'abstain_weight': abstain_weight, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Fixed eight-word vocabulary and manually authored synthetic examples.',
                              'No entity recognition, address selection, writes, correction, or persistence.',
                              'Held templates and names are a small development set, not an open-world safety evaluation.',
                              'A predicted value alone does not establish source reliability or authorization to write.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({name: result['summary'] for name, result in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--augmented', action='store_true')
    parser.add_argument('--diverse', action='store_true')
    parser.add_argument('--abstain-weight', type=float, default=1.0)
    args = parser.parse_args()
    run(args.output, args.augmented, args.diverse, args.abstain_weight)
