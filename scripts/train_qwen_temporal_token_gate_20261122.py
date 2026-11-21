"""Compare pooled and token-attentive time classification on a frozen SPORA checkpoint."""

import argparse
import json
import time
from pathlib import Path

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_temporal_contrast_cases import (
    ABSTAIN, CAL_NON_READ, CAL_TRIPLES, NON_READ, VERSION_TRIPLES, temporal_rows,
)

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.named_address import feature_files_sha256
from spora.research_temporal_intent import (
    FactorizedTemporalGate, TemporalTokenGate, question_mean, temporal_logits, time_selection,
)

CLASSES = ('current', 'past', 'later', 'abstain')
INDICES = (6, 18)
SEED = 20261122


def suites():
    from scripts.audit_qwen_compatible_natural_memory_20261121 import cases, fresh_cases, fresh_version_cases

    train_names = ('顾岚', '韩青', '刘芷', '安渝', '叶雨桐', '周乔', '陈北', '苏澄', '郭璇', '许墨', '王景', '何芸')
    fields = ('饮品', '常用称呼', '送达地点', '字体', '饮料', '出发地点', '食谱', '偏好颜色')
    train = temporal_rows(train_names, fields, VERSION_TRIPLES, NON_READ)
    calibration = temporal_rows(('任湘', '江淼', '吴昕', '张平'), ('饮料', '尺码'), CAL_TRIPLES, CAL_NON_READ)

    def existing(rows):
        return [(question, ABSTAIN if label is None else label) for question, _, label in rows]

    return {'train': train, 'calibration': calibration, 'original': existing(cases()),
            'fresh_v1': existing(fresh_cases()), 'fresh_v2': existing(fresh_version_cases())}


def fit(architecture, features, masks, targets):
    width = features.shape[1] * features.shape[-1]
    if architecture == 'mean':
        head = nn.Linear(width, len(CLASSES))
        inputs = (question_mean(features, masks),)
    elif architecture == 'factorized':
        head = FactorizedTemporalGate(width)
        inputs = (question_mean(features, masks),)
    else:
        head = TemporalTokenGate(width, len(CLASSES))
        inputs = (features, masks)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.02)
    generator = torch.Generator().manual_seed(SEED)
    for _ in range(12):
        for indices in torch.randperm(len(targets), generator=generator).split(24):
            optimizer.zero_grad(set_to_none=True)
            logits = head(*(value[indices] for value in inputs))
            loss = F.cross_entropy(logits, targets[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite temporal training loss')
            loss.backward()
            optimizer.step()
    return head.eval().state_dict()


def threshold_for(probabilities, targets, *, factorized):
    initial = .5 if factorized else 0.
    winner = time_selection(probabilities, initial, factorized=factorized)
    confidence = probabilities[:, :3].sum(-1) if factorized else probabilities.max(-1).values
    wrong_reads = confidence[(targets == ABSTAIN) & (winner != ABSTAIN)]
    if not len(wrong_reads):
        return initial
    return float(torch.nextafter(wrong_reads.max(), torch.tensor(float('inf'))))


def evaluate(rows, probabilities, threshold, *, factorized):
    predicted = time_selection(probabilities, threshold, factorized=factorized).tolist()
    details = [{'question': question, 'label': label, 'prediction': guess}
               for (question, label), guess in zip(rows, predicted, strict=True)]
    by_label = {}
    for index, label in enumerate(CLASSES):
        guesses = [row['prediction'] for row in details if row['label'] == index]
        by_label[label] = {'count': len(guesses), 'correct': guesses.count(index),
                           'abstained': guesses.count(ABSTAIN)}
    return {'by_label': by_label, 'rows': details}


def main(checkpoint, baseline, output, architectures):
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    old_path = baseline / 'temporal.pt'
    old_report = json.loads((baseline / 'result.json').read_text(encoding='utf-8'))
    checkpoint_sha, baseline_sha = sha(checkpoint), sha(old_path)
    feature_sha = feature_files_sha256(MODEL)
    old = torch.load(old_path, map_location='cpu', weights_only=True)
    if (baseline_sha != old_report['temporal_sha256'] or old['checkpoint_sha256'] != checkpoint_sha
            or old['base_sha256'] != WEIGHT_SHA or old['feature_config_sha256'] != feature_sha
            or tuple(old['hidden_indices']) != INDICES or tuple(old['classes']) != CLASSES):
        raise ValueError('Baseline time head provenance mismatch')
    data = suites()
    boxes = {}
    questions = []
    for name, rows in data.items():
        boxes[name] = slice(len(questions), len(questions) + len(rows))
        questions.extend(question for question, _ in rows)
    if len(set(questions)) != len(questions):
        raise ValueError('Repeated question across temporal splits')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    decoder, _, _ = load_integrated(checkpoint, base, MODEL)
    decoder.eval().requires_grad_(False)
    features, masks = extract(decoder.base, tokenizer, questions, batch_size=2, hidden_indices=INDICES,
                              last_logit_only=True)
    del decoder, base
    targets = torch.tensor([label for rows in data.values() for _, label in rows])
    models = {'original': old}
    for architecture in architectures:
        training = boxes['train']
        models[architecture] = {'format': 'spora-bounded-temporal-gate-v2', 'architecture': architecture,
                                'base_sha256': WEIGHT_SHA, 'checkpoint_sha256': checkpoint_sha,
                                'feature_config_sha256': feature_sha, 'hidden_indices': list(INDICES),
                                'classes': list(CLASSES), 'state': fit(architecture, features[training],
                                                                     masks[training], targets[training])}
    if sha(checkpoint) != checkpoint_sha or sha(old_path) != baseline_sha:
        raise ValueError('Protected time experiment inputs changed')
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in models.items():
        with torch.inference_mode():
            probabilities = torch.cat([temporal_logits(payload, x, mask).softmax(-1)
                                       for x, mask in zip(features.split(24), masks.split(24), strict=True)])
        calibration = boxes['calibration']
        factorized = payload.get('architecture') == 'factorized'
        threshold = (old['threshold'] if name == 'original' else
                     threshold_for(probabilities[calibration], targets[calibration], factorized=factorized))
        evaluated = {split: evaluate(rows, probabilities[boxes[split]], threshold, factorized=factorized)
                     for split, rows in data.items() if split != 'train'}
        folder = output / name
        folder.mkdir()
        weights = folder / 'temporal.pt'
        atomic_save(dict(payload, threshold=threshold), weights)
        train_count = old_report.get('train_count') if name == 'original' else len(data['train'])
        report = {'temporal_sha256': sha(weights), 'checkpoint_sha256': checkpoint_sha,
                  'baseline_sha256': baseline_sha, 'threshold': threshold, 'train_count': train_count,
                  'seconds': time.perf_counter() - start, 'results': evaluated}
        (folder / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(name, threshold, {split: rows['by_label'] for split, rows in evaluated.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--architectures', nargs='+', choices=('mean', 'token_attention', 'factorized'),
                        default=('mean', 'token_attention', 'factorized'))
    args = parser.parse_args()
    main(args.checkpoint, args.baseline, args.output, args.architectures)
