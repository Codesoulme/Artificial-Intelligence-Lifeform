"""Develop a full-utterance near-name identity gate, never used for writes."""
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
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import NAMES
from scripts.deepseek_observation_value_probe import pool

OUTPUT = ROOT / 'runs/deepseek-spora-identity-gate-probe-20261111'

# Suffixes and wording are split to expose transfer failures. Known identities
# must necessarily be shared across the enrollment and evaluation partitions.
SPLITS = {
    'train': (('{n}现在喜欢喝什么？', '请查一下{n}记录的饮料。'), ('哥', '姐', '仔', '叔')),
    'calibration': (('{n}目前的饮品偏好是什么？',), ('子', '娜')),
    'held': (('{n}现在最爱喝什么饮料？', '说说{n}喜欢的饮品。'), ('林', '宁', '明')),
}


def build(split):
    templates, suffixes = SPLITS[split]
    rows = []
    for name in NAMES:
        for template in templates:
            rows.append({'text': template.format(n=name), 'label': 1, 'name': name})
            for suffix in suffixes:
                rows.append({'text': template.format(n=name + suffix), 'label': 0, 'name': name + suffix})
    return rows


def evaluate(model, vectors, rows, threshold):
    with torch.no_grad():
        probability = model(vectors).squeeze(-1).sigmoid()
    output = [{**row, 'score': float(p), 'accepted': float(p) >= threshold} for row, p in zip(rows, probability)]
    return {'known_total': sum(r['label'] == 1 for r in output),
            'known_accepted': sum(r['label'] == 1 and r['accepted'] for r in output),
            'unknown_total': sum(r['label'] == 0 for r in output),
            'unknown_accepted': sum(r['label'] == 0 and r['accepted'] for r in output), 'rows': output}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = {split: build(split) for split in SPLITS}
    for left in rows:
        for right in rows:
            if left != right and {r['text'] for r in rows[left]} & {r['text'] for r in rows[right]}:
                raise ValueError('Split text overlap')
    checkpoint_hash = runtime.sha(runtime.CHECKPOINT)
    torch.manual_seed(90121)
    torch.set_num_threads(2)
    started = time.perf_counter()
    # A nonexistent temporary working path ensures no real individual is opened.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        vectors = {}
        for split, examples in rows.items():
            features, masks = runtime.extract(chat.base, chat.tokenizer, [r['text'] for r in examples], batch_size=2)
            vectors[split] = pool(features, masks).float()
            print(split, len(examples), flush=True)
        model = nn.Linear(vectors['train'].shape[-1], 1)
        labels = torch.tensor([r['label'] for r in rows['train']], dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
        for _ in range(400):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(vectors['train']).squeeze(-1), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            cal_scores = model(vectors['calibration']).sigmoid().squeeze(-1)
        thresholds = sorted({float(v) for v in cal_scores}) + [1.01]
        candidates = [(evaluate(model, vectors['calibration'], rows['calibration'], t), t) for t in thresholds]
        feasible = [(result, t) for result, t in candidates if result['unknown_accepted'] == 0]
        selected, threshold = max(feasible, key=lambda pair: (pair[0]['known_accepted'], -pair[1]))
        results = {split: evaluate(model, vectors[split], examples, threshold) for split, examples in rows.items()}
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_hash:
        raise ValueError('Bootstrap changed')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    path = OUTPUT / 'gate.pt'
    torch.save({'format': 'spora-identity-gate-development-v1', 'state': model.state_dict(),
                'threshold': threshold, 'source_checkpoint_sha256': checkpoint_hash}, path)
    report = {'status': 'research_only_no_writes', 'threshold': threshold, 'checkpoint_sha256': runtime.sha(path),
              'source_checkpoint_sha256': checkpoint_hash, 'results': results,
              'seconds': time.perf_counter() - started,
              'limitations': ['Synthetic names, suffixes, and templates; not an open-world identity verifier.',
                              'Same known identities appear in all partitions; no new-identity enrollment.',
                              'Threshold chosen on small calibration negatives; zero errors is not a safety guarantee.',
                              'Uses whole utterance with causal hidden features; cannot resolve homonyms.',
                              'Not integrated into default routing; no memory writes.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, result in results.items():
        print(split, {k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
