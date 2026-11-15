"""Calibrate per-class abstention and report acceptance/error tradeoffs offline."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_relation_selection_probe import LABELS

HEAD = ROOT / 'runs/deepseek-spora-relation-selection-probe-20261116/head.pt'
PROBE = ROOT / 'runs/deepseek-spora-relation-selection-probe-20261116/result.json'
STRESS = ROOT / 'runs/deepseek-spora-relation-selection-stress-20261116.json'
COMPOSE = ROOT / 'runs/deepseek-spora-relation-multifact-composition-20261116.json'
OUT = ROOT / 'runs/deepseek-spora-relation-selective-audit-20261116.json'


def thresholds_from_calibration(scores, labels):
    """Set each threshold strictly above the maximum observed wrong-label score."""
    if scores.ndim != 2 or scores.shape[1] != len(LABELS) or len(labels) != scores.shape[0]:
        raise ValueError('Invalid calibration inputs')
    out = {}
    for index, label in enumerate(LABELS[:-1]):
        negative = scores[torch.tensor([value != label for value in labels]), index]
        if not len(negative) or not bool(torch.isfinite(negative).all()):
            raise ValueError('Calibration requires finite negatives')
        maximum = negative.max()
        out[label] = float(torch.nextafter(maximum, torch.tensor(float('inf'), dtype=maximum.dtype)))
    return out


def decide(scores, thresholds):
    if scores.ndim != 1 or scores.numel() != len(LABELS) or not bool(torch.isfinite(scores).all()):
        raise ValueError('Invalid class scores')
    index = int(scores.argmax())
    label = LABELS[index]
    return label if label != 'reject' and float(scores[index]) >= thresholds[label] else 'reject'


def summarize(rows):
    relevant = [r for r in rows if r['expected'] != 'reject']
    reject = [r for r in rows if r['expected'] == 'reject']
    return {'cases': len(rows), 'correct': sum(r['expected'] == r['decision'] for r in rows),
            'positive_cases': len(relevant), 'positive_correct': sum(r['decision'] == r['expected'] for r in relevant),
            'false_reject': sum(r['decision'] == 'reject' for r in relevant),
            'reject_cases': len(reject), 'reject_misread': sum(r['decision'] != 'reject' for r in reject)}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    protected = {path: p.sha(path) for path in (HEAD, PROBE, STRESS, COMPOSE,
                 p.MODEL / 'model.safetensors', p.runtime.CHECKPOINT)}
    data = torch.load(HEAD, map_location='cpu', weights_only=True)
    if (protected[p.MODEL / 'model.safetensors'] != p.WEIGHT_SHA
            or data['base_sha256'] != p.WEIGHT_SHA
            or data['checkpoint_sha256'] != protected[p.runtime.CHECKPOINT]
            or tuple(data['labels']) != LABELS):
        raise ValueError('Probe lineage mismatch')
    probe = json.loads(PROBE.read_text(encoding='utf-8'))
    stress = json.loads(STRESS.read_text(encoding='utf-8'))
    composition = json.loads(COMPOSE.read_text(encoding='utf-8'))
    items = ([{'group': row['split'], 'question': row['question'], 'expected': row['label']}
              for row in probe['rows'] if row['split'] in ('dev', 'test')]
             + [{'group': 'stress', 'question': row['question'], 'expected': row['expected']}
                for row in stress['rows']]
             + [{'group': 'composition', 'question': row['question'], 'expected': row['expected_relation']}
                for row in composition['before']])
    if len({(row['group'], row['question']) for row in items}) != len(items):
        raise ValueError('Duplicate questions within an evaluation group')
    torch.set_num_threads(2)
    head = nn.Linear(data['state']['weight'].shape[1], len(LABELS))
    head.load_state_dict(data['state'], strict=True)
    with tempfile.TemporaryDirectory(prefix='spora-selective-') as temp:
        p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original = p.frozen_digest(chat.organism, chat.state)
        features, masks = extract(chat.base, chat.tokenizer, [row['question'] for row in items], batch_size=1)
        with torch.inference_mode():
            probabilities = head(pool(features, masks).float()).softmax(-1)
        if p.frozen_digest(chat.organism, chat.state) != original:
            raise ValueError('Individual changed')
    calibration_indices = [i for i, row in enumerate(items) if row['group'] == 'dev']
    thresholds = thresholds_from_calibration(probabilities[calibration_indices],
                                              [items[i]['expected'] for i in calibration_indices])
    for row, scores in zip(items, probabilities):
        row['probabilities'] = scores.tolist()
        row['baseline'] = LABELS[int(scores.argmax())]
        row['decision'] = decide(scores, thresholds)
    summaries = {}
    for group in ('dev', 'test', 'stress', 'composition'):
        group_rows = [row for row in items if row['group'] == group]
        summaries[group] = {'selective': summarize(group_rows),
                            'baseline': summarize([dict(row, decision=row['baseline']) for row in group_rows])}
    if any(p.sha(path) != digest for path, digest in protected.items()):
        raise ValueError('Protected source changed')
    report = {'thresholds': thresholds, 'summaries': summaries, 'rows': items,
              'head_sha256': protected[HEAD], 'base_sha256': p.WEIGHT_SHA,
              'limitations': ['Calibration uses only 20 authored examples; no distributional guarantee.',
                              'Dev, stress and composition outputs have been examined in prior work.',
                              'Class probabilities are not calibrated confidence or identity verification.',
                              'No actual memory access, natural generation, or runtime deployment.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'thresholds': thresholds, 'summaries': summaries}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
