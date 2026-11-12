"""Retrain only polarity and value-span heads for broader Chinese wording."""
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
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from scripts.deepseek_spora_value_span_probe import SpanHead, feature_batch, decode as decode_values
from scripts.deepseek_spora_value_span_probe import labels as value_labels
from scripts.deepseek_spora_polarity_gate_probe import CLASSES, pool as polarity_pool
from scripts.deepseek_observation_value_probe import WORDS
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement


OUTPUT = ROOT / 'runs/deepseek-spora-natural-write-coverage-retrain-20261113'
POSITIVE = (
    '{n}喜欢喝{w}。', '{n}最喜欢的饮料是{w}。', '我知道{n}爱喝{w}。',
    '{n}平时最爱喝{w}。', '{n}日常更愿意选择喝{w}。',
    '{n}通常会选择{w}作为饮料。', '{n}偏爱的饮品是{w}。',
)
NEGATIVE = (
    ('negation', ('{n}不喜欢喝{w}。', '{n}从来不喝{w}。', '{n}一点也不爱喝{w}。')),
    ('question', ('你知道{n}喜欢喝什么吗？', '{n}是否爱喝{w}？')),
    ('hypothesis', ('如果{n}喜欢喝{w}就好了。', '也许{n}爱喝{w}，我不确定。')),
    ('third_party', ('{n}说阿远喜欢喝{w}。', '听说{n}可能喜欢{w}。')),
    ('action', ('{n}给阿远买了{w}。', '{n}为同事准备了{w}。')),
)
NAMES = {'train': ('小林', '阿宁', '小雨'),
         'calibration': ('阿哲',), 'held': ('小梅', '阿远')}


def build(split):
    rows = []
    for name in NAMES[split]:
        for word in WORDS:
            for template in POSITIVE:
                start = template.format(n=name, w=word).index(word)
                text = template.format(n=name, w=word)
                rows.append({'text': text, 'start': start, 'end': start + len(word),
                             'positive': True, 'kind': 'affirmative'})
            for kind, templates in NEGATIVE:
                for template in templates:
                    rows.append({'text': template.format(n=name, w=word),
                                 'start': None, 'end': None, 'positive': False,
                                 'kind': kind})
    return rows


def train_value(vectors, masks, offsets, starts, rows, steps=600):
    model = SpanHead(vectors.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    accepted, boundaries = value_labels(rows, offsets, starts, masks)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        accept_logits, boundary_logits = model(vectors)
        active = masks
        loss = F.binary_cross_entropy_with_logits(accept_logits, accepted)
        loss = loss + F.binary_cross_entropy_with_logits(boundary_logits[active], boundaries[active])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval()


def train_polarity(vectors, rows, steps=600):
    model = nn.Linear(vectors.shape[-1], len(CLASSES))
    aliases = {'question': 'question_history'}
    labels = torch.tensor([CLASSES.index(aliases.get(row['kind'], row['kind'])) for row in rows])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(vectors), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model.eval()


def evaluate(rows, vectors, masks, offsets, starts, value, polarity, value_threshold, polarity_threshold):
    value_rows = decode_values(rows, value, vectors, offsets, starts, masks, value_threshold)
    with torch.inference_mode():
        probabilities = polarity(polarity_pool(vectors, masks)).softmax(-1)
    result = []
    for row, value_row, probs in zip(rows, value_rows, probabilities):
        prediction = int(probs.argmax())
        polarity_ok = prediction == 0 and float(probs[0]) >= polarity_threshold
        value = value_row['value']
        value_ok = bool(value_row['accepted'] and value in WORDS)
        authorized = polarity_ok and value_ok
        result.append({**row, 'value': value, 'value_exact': bool(row['positive'] and value == row['text'][row['start']:row['end']]),
                       'value_ok': value_ok, 'polarity': CLASSES[prediction],
                       'polarity_probability': float(probs[0]), 'polarity_ok': polarity_ok,
                       'authorized': authorized})
    positives = [row for row in result if row['positive']]
    negatives = [row for row in result if not row['positive']]
    return {'positive_value_exact': sum(row['value_exact'] for row in positives),
            'positive_total': len(positives),
            'positive_authorized': sum(row['authorized'] for row in positives),
            'negative_authorized': sum(row['authorized'] for row in negatives),
            'negative_total': len(negatives), 'rows': result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.manual_seed(93113)
    torch.set_num_threads(2)
    started = time.perf_counter()
    source = {str(MODEL / 'model.safetensors'): sha(MODEL / 'model.safetensors'),
              str(CHECKPOINT): sha(CHECKPOINT)}
    if source[str(MODEL / 'model.safetensors')] != WEIGHT_SHA:
        raise ValueError('Base weights changed')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    del decoder, organism, state
    blocks = {i: layer for i, layer in enumerate(base.model.layers)
              if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    datasets = {split: build(split) for split in NAMES}
    tensors = {}
    for split in ('train',):
        rows = datasets[split]
        vectors, masks, offsets, starts = feature_batch(base, tokenizer, rows)
        tensors[split] = (vectors, masks, offsets, starts)
        print(split, len(rows), flush=True)
    train_vectors = polarity_pool(tensors['train'][0], tensors['train'][1])
    polarity = train_polarity(train_vectors, datasets['train'])
    value = train_value(*tensors['train'], datasets['train'])
    del train_vectors, tensors['train']
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    for split in ('calibration', 'held'):
        rows = datasets[split]
        vectors, masks, offsets, starts = feature_batch(base, tokenizer, rows)
        tensors[split] = (vectors, masks, offsets, starts)
        print(split, len(rows), flush=True)
    cal_vectors = polarity_pool(tensors['calibration'][0], tensors['calibration'][1])
    with torch.inference_mode():
        cal_polarity = polarity(cal_vectors).softmax(-1)[:, 0]
    polarity_thresholds = sorted(set(float(x) for x in cal_polarity)) + [1.01]
    value_thresholds = [float(x) for x in torch.linspace(-2.0, 2.0, 41)]
    choices = []
    for p_threshold in polarity_thresholds:
        for v_threshold in value_thresholds:
            result = evaluate(datasets['calibration'], *tensors['calibration'], value, polarity,
                              v_threshold, p_threshold)
            if result['negative_authorized'] == 0:
                choices.append((result['positive_authorized'], result['positive_value_exact'],
                                -p_threshold, -v_threshold, p_threshold, v_threshold))
    if not choices:
        raise ValueError('No zero-false-authorization calibration threshold')
    choice = max(choices)
    _, _, _, _, polarity_threshold, value_threshold = choice
    train_features = feature_batch(base, tokenizer, datasets['train'])
    tensors['train'] = train_features
    results = {split: evaluate(datasets[split], *tensors[split], value, polarity,
                               value_threshold, polarity_threshold)
               for split in datasets}
    if any(sha(Path(path)) != digest for path, digest in source.items()):
        raise ValueError('Protected source changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    candidate = OUTPUT / 'heads.pt'
    torch.save({'format': 'spora-natural-write-coverage-heads-v1',
                'polarity': polarity.state_dict(), 'value': value.state_dict(),
                'polarity_threshold': polarity_threshold, 'value_threshold': value_threshold,
                'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': source[str(CHECKPOINT)],
                'feature_variant_sha256': variant}, candidate)
    report = {'status': 'research_only_no_writes', 'thresholds': {'polarity': polarity_threshold,
              'value': value_threshold}, 'source_hashes': source,
              'feature_variant_sha256': variant, 'results': results,
              'steps': {'polarity': 600, 'value': 600},
              'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic templates and finite beverage vocabulary.',
                              'Entity router was frozen and not re-evaluated here.',
                              'Zero calibration false authorization is not an open-domain guarantee.',
                              'Candidate heads are not connected to the default runtime.'],
              'candidate_sha256': sha(candidate)}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({split: {key: value[key] for key in ('positive_value_exact', 'positive_total',
                    'positive_authorized', 'negative_authorized', 'negative_total')}
                      for split, value in results.items()}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
