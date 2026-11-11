"""Compare separately trained fact/value heads on replaced model features."""
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
from scripts.deepseek_spora_observation_retrain import SPLITS, build
from scripts.deepseek_spora_observation_transfer_audit import CHECKPOINT
from spora.integrated_checkpoint import load_integrated, replacement_feature_variant_sha256
from spora.structure_transfer import SPORALayerReplacement

PRIOR = ROOT / 'runs/deepseek-spora-observation-retrain-20261111/classifier.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-observation-separate-heads-20261111'
UNKNOWN = {'train': ('茉莉花茶', '苏打水', '绿豆汤'),
           'calibration': ('热巧克力', '冰红茶'),
           'held': ('柠檬水', '苹果汁', '白开水', '汽水')}


def dataset(split):
    rows = build(split)
    names, positives, _ = SPLITS[split]
    rows += [{'text': pattern.format(n=name, w=word), 'group': 'unknown', 'label': ABSTAIN}
             for name in names for word in UNKNOWN[split] for pattern in positives]
    return rows


def predict(model, vectors, rows, threshold):
    with torch.inference_mode():
        fact, values = model(vectors)
        value_prediction = values.argmax(-1)
        output = torch.where(fact >= threshold, value_prediction, torch.full_like(value_prediction, ABSTAIN))
    records = [{**r, 'prediction': int(p), 'value_prediction': int(v), 'fact_logit': float(f)}
               for r, p, v, f in zip(rows, output, value_prediction, fact)]
    summary = {}
    for group in ('positive', 'negative', 'unknown'):
        selected = [r for r in records if r['group'] == group]
        summary[group] = {'total': len(selected), 'correct': sum(r['label'] == r['prediction'] for r in selected),
                          'accepted': sum(r['prediction'] != ABSTAIN for r in selected)}
    return {'summary': summary, 'rows': records}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = (MODEL / 'model.safetensors', CHECKPOINT, PRIOR)
    original_hashes = {str(p): sha(p) for p in paths}
    if original_hashes[str(paths[0])] != WEIGHT_SHA:
        raise ValueError('Unexpected base model')
    sets = {split: dataset(split) for split in SPLITS}
    for a in sets:
        for b in sets:
            if a != b and {r['text'] for r in sets[a]} & {r['text'] for r in sets[b]}:
                raise ValueError('Overlapping splits')
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
    for split, rows in sets.items():
        features, masks = extract(base, tokenizer, [r['text'] for r in rows], batch_size=2)
        vectors[split] = pool(features, masks)
        del features, masks
        print(split, len(rows), flush=True)
    old_data = torch.load(PRIOR, map_location='cpu', weights_only=True)
    if old_data['feature_variant_sha256'] != variant:
        raise ValueError('Prior probe feature variant mismatch')
    old = FactorizedProbe(vectors['train'].shape[-1]).eval()
    old.load_state_dict(old_data['state'], strict=True)
    model = FactorizedProbe(vectors['train'].shape[-1])
    fact_targets = torch.tensor([r['group'] != 'negative' for r in sets['train']], dtype=torch.float32)
    value_mask = torch.tensor([r['group'] != 'negative' for r in sets['train']])
    value_targets = torch.tensor([r['label'] for r in sets['train']])[value_mask]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    for _ in range(400):
        optimizer.zero_grad(set_to_none=True)
        facts, values = model(vectors['train'])
        loss = F.binary_cross_entropy_with_logits(facts, fact_targets) + F.cross_entropy(values[value_mask], value_targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        facts, _ = model(vectors['calibration'])
    thresholds = sorted(set(float(x) for x in facts))
    thresholds.append(thresholds[-1] + 1.0)
    choices = [(predict(model, vectors['calibration'], sets['calibration'], threshold), threshold)
               for threshold in thresholds]
    feasible = [(result, threshold) for result, threshold in choices
                if result['summary']['negative']['accepted'] == 0 and result['summary']['unknown']['accepted'] == 0]
    best, threshold = max(feasible, key=lambda pair: (pair[0]['summary']['positive']['correct'], -pair[1]))
    results = {split: {'prior': predict(old, vectors[split], rows, old_data['threshold']),
                       'separate': predict(model, vectors[split], rows, threshold)}
               for split, rows in sets.items()}
    if any(sha(p) != original_hashes[str(p)] for p in paths):
        raise ValueError('Protected artifacts changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / 'classifier.pt'
    torch.save({'format': 'spora-separate-observation-development-v1', 'state': model.state_dict(),
                'threshold': threshold, 'base_sha256': WEIGHT_SHA, 'source_checkpoint_sha256': original_hashes[str(CHECKPOINT)],
                'feature_variant_sha256': variant, 'vocabulary': WORDS}, checkpoint)
    report = {'status': 'research_only_no_writes', 'threshold': threshold,
              'selection': 'zero observed calibration negative and unknown acceptances; maximize correct positives',
              'checkpoint_sha256': sha(checkpoint), 'source_hashes': original_hashes,
              'feature_variant_sha256': variant, 'seconds': time.perf_counter() - started,
              'results': results, 'limitations': ['Synthetic templates, no independent real dialogue.',
              'Unknown examples are finite and share templates with known values.',
              'No entity/source routing or real memory writing.',
              'Calibration cannot guarantee zero errors on new data.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for name, result in results.items():
        print(name, {key: item['summary'] for key, item in result.items()}, flush=True)


if __name__ == '__main__':
    main()
