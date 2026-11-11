"""Develop an isolated finite-value observation probe for replaced features."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_observation_factorized_probe import FactorizedProbe
from scripts.deepseek_observation_value_probe import ABSTAIN, WORDS, pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT, PROBE
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

OUTPUT = ROOT / 'runs/deepseek-spora-observation-retrain-20261111'

# Split by both person and wording; all texts remain synthetic development data.
SPLITS = {
    'train': (('小林', '阿宁', '小雨'),
              ('{n}喜欢喝{w}。', '{n}最喜欢的饮料是{w}。', '我知道{n}爱喝{w}。'),
              ('{n}不喜欢喝{w}。', '{n}喜欢喝{w}吗？', '假如{n}爱喝{w}就好了。',
               '{n}过去喜欢{w}，现在不知道。', '{n}给小周买了{w}。',
               '{n}说小周喜欢喝{w}。', '{n}觉得{w}难喝。',
               '我猜{n}可能喜欢{w}。')),
    'calibration': (('阿哲',),
              ('{n}最中意喝{w}。', '{n}平时最爱喝{w}。'),
              ('{n}是否爱喝{w}？', '{n}从来不喝{w}。', '{n}以前爱喝{w}，现状不明。',
               '{n}替朋友购买了{w}。', '听说{name}可能喜欢{w}。'.replace('{name}', '{n}'),
               '{n}说小周爱喝{w}。')),
    'held': (('小梅', '阿远'),
              ('据我所知，{n}最喜欢的饮品是{w}。', '{n}偏爱的饮料是{w}。'),
              ('{n}压根不爱喝{w}。', '你知道{n}喜欢喝{w}吗？',
               '如果{n}喜欢喝{w}就好了。', '{n}曾经喜欢{w}，后来怎样不清楚。',
               '{n}为同事准备了{w}。', '{n}猜自己也许喜欢{w}。',
               '{n}说阿明最喜欢喝{w}。', '{n}不再喜欢喝{w}了。')),
}


def build(split):
    names, positives, negatives = SPLITS[split]
    return [{'text': template.format(n=name, w=word), 'group': group, 'label': label}
            for name in names for label, word in enumerate(WORDS)
            for group, templates in (('positive', positives), ('negative', negatives))
            for template in templates
            for label in ([label] if group == 'positive' else [ABSTAIN])]


def evaluate(model, vectors, rows, threshold):
    with torch.inference_mode():
        fact, logits = model(vectors)
        labels = logits.argmax(-1)
        predictions = torch.where(fact >= threshold, labels, torch.full_like(labels, ABSTAIN))
    result = [{**row, 'prediction': int(pred), 'fact_logit': float(score)}
              for row, pred, score in zip(rows, predictions, fact)]
    positives = [r for r in result if r['group'] == 'positive']
    negatives = [r for r in result if r['group'] == 'negative']
    return {'positive_correct': sum(r['prediction'] == r['label'] for r in positives),
            'positive_total': len(positives),
            'negative_false_write': sum(r['prediction'] != ABSTAIN for r in negatives),
            'negative_total': len(negatives), 'rows': result}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = [MODEL / 'model.safetensors', CHECKPOINT, PROBE]
    before = {str(p): sha(p) for p in files}
    if before[str(files[0])] != WEIGHT_SHA:
        raise ValueError('Unexpected base weights')
    datasets = {name: build(name) for name in SPLITS}
    for left in datasets:
        for right in datasets:
            if left != right and {r['text'] for r in datasets[left]} & {r['text'] for r in datasets[right]}:
                raise ValueError('Overlapping text across splits')
    torch.manual_seed(30112)
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    decoder, organism, state = load_integrated(CHECKPOINT, base, MODEL, individual_device='cuda')
    blocks = {i: layer for i, layer in enumerate(base.model.layers) if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(MODEL, blocks)
    vectors = {}
    for name, rows in datasets.items():
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        vectors[name] = pool(features, masks)
        del features, masks
        print(name, len(rows), flush=True)
    previous_data = torch.load(PROBE, map_location='cpu', weights_only=True)
    previous = FactorizedProbe(vectors['train'].shape[-1]).eval()
    previous.load_state_dict(previous_data['state'], strict=True)
    candidate = FactorizedProbe(vectors['train'].shape[-1])
    fact_targets = torch.tensor([r['group'] == 'positive' for r in datasets['train']], dtype=torch.float32)
    labels = torch.tensor([r['label'] for r in datasets['train']])
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.0002, weight_decay=0.01)
    for step in range(400):
        optimizer.zero_grad(set_to_none=True)
        facts, values = candidate(vectors['train'])
        loss = F.binary_cross_entropy_with_logits(facts, fact_targets) + F.cross_entropy(values, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    candidate.eval()
    # Pick a threshold only on calibration; ties favor positive coverage.
    fact_scores = candidate(vectors['calibration'])[0].detach()
    thresholds = sorted(set(float(x) for x in fact_scores))
    thresholds += [thresholds[-1] + 1.0]
    candidates = [(evaluate(candidate, vectors['calibration'], datasets['calibration'], t), t)
                  for t in thresholds]
    feasible = [(r, t) for r, t in candidates if r['negative_false_write'] == 0]
    chosen, threshold = max(feasible, key=lambda rt: (rt[0]['positive_correct'], -rt[1]))
    results = {name: {'previous': evaluate(previous, vectors[name], rows, 0.0),
                      'candidate': evaluate(candidate, vectors[name], rows, threshold)}
               for name, rows in datasets.items()}
    if any(sha(p) != before[str(p)] for p in files):
        raise ValueError('Protected artifact changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'classifier.pt'
    torch.save({'format': 'spora-observation-replacement-development-v1',
                'state': candidate.state_dict(), 'threshold': threshold,
                'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': before[str(CHECKPOINT)],
                'feature_variant_sha256': variant, 'vocabulary': WORDS}, checkpoint)
    report = {'status': 'research_only_no_writes', 'threshold': threshold,
              'calibration_selection': 'zero observed false writes, maximize correct positives',
              'checkpoint_sha256': sha(checkpoint), 'source_hashes': before,
              'feature_variant_sha256': variant, 'replacement_layers': sorted(blocks),
              'steps': 400, 'seconds': time.perf_counter() - started, 'results': results,
              'limitations': ['Synthetic small-scale templates, not independent real dialogue.',
                              'Threshold is selected on finite calibration data, not a false-write guarantee.',
                              'Only eight values; no entity, attribution, source trust, or actual writes.',
                              'No unsupported vocabulary coverage in this run.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, outcome in results.items():
        print(name, {k: {m: v[m] for m in ('positive_correct', 'positive_total', 'negative_false_write', 'negative_total')}
                     for k, v in outcome.items()}, flush=True)


if __name__ == '__main__':
    main()
