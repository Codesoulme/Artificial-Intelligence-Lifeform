"""Evaluate an independent Chinese name tagger as a source of training evidence."""

import argparse
import json
from pathlib import Path

import torch

from scripts.audit_qwen_compatible_natural_memory_20261121 import sha
from scripts.audit_registered_name_spans import metrics
from transformers import AutoModelForTokenClassification, AutoTokenizer


TEACHER = 'shibing624/bert4ner-base-chinese'
REVISION = '5d660ed2aa9da482bf2d99c6bc8cf2ce66758f6a'
WEIGHTS_SHA = '1d2f0ea479431c1907bbaa75f872c26267ce3c19ea78e4266980b0d49b5a825a'


def bio_transitions(labels):
    allowed = torch.ones(len(labels), len(labels), dtype=torch.bool)
    first = torch.ones(len(labels), dtype=torch.bool)
    for current, label in enumerate(labels):
        if label.startswith('I-'):
            entity = label[2:]
            first[current] = False
            allowed[:, current] = torch.tensor([previous in ('B-' + entity, label) for previous in labels])
    return torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, -torch.inf), first


def viterbi(logits, transitions, first):
    scores = logits[0].masked_fill(~first, -torch.inf)
    back = []
    for token in logits[1:]:
        previous_score, previous_choice = (scores[:, None] + transitions).max(0)
        scores = previous_score + token
        back.append(previous_choice)
    path = [int(scores.argmax())]
    for choices in reversed(back):
        path.append(int(choices[path[-1]]))
    return list(reversed(path))


def person_spans(offsets, path, labels):
    spans = []
    start = end = None
    for (left, right), index in zip(offsets, path, strict=True):
        label = labels[index]
        if start is not None and (label != 'I-PER' or end != left):
            spans.append([start, end])
            start = end = None
        if label == 'B-PER':
            start, end = left, right
        elif label == 'I-PER' and start is not None:
            end = right
    if start is not None:
        spans.append([start, end])
    return spans


@torch.inference_mode()
def annotate(model, tokenizer, texts):
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if not {'O', 'B-PER', 'I-PER'}.issubset(labels) or not tokenizer.is_fast:
        raise ValueError('Expected a token classifier with BIO person labels and original offsets')
    transitions, first = bio_transitions(labels)
    for batch_start in range(0, len(texts), 16):
        batch = texts[batch_start:batch_start + 16]
        inputs = tokenizer(batch, padding=True, return_offsets_mapping=True, return_tensors='pt')
        offsets = inputs.pop('offset_mapping')
        logits = model(**inputs).logits.float()
        for text, off, scores in zip(batch, offsets, logits, strict=True):
            left, right = off.unbind(-1)
            valid = right > left
            if not bool(valid.any()) or bool((right > len(text)).any()):
                raise ValueError('Invalid token offsets')
            off, scores = off[valid].tolist(), scores[valid]
            yield {'greedy': person_spans(off, scores.argmax(-1).tolist(), labels),
                   'bio': person_spans(off, viterbi(scores, transitions, first), labels)}


def compare(rows):
    for row in rows:
        spans = row['candidates']
        for search in ('greedy', 'bio'):
            detected = row['teacher'][search]
            same = len(spans) == 1 and spans[0][:2] in detected
            read = same and row.get('temporal_mode', 3) != 3
            row[search + '_exact'] = row['unique'] if read else None
            row[search + '_single'] = row['unique'] if read and len(detected) == 1 else None
    return {path: metrics(rows, path) for path in ('greedy_exact', 'greedy_single', 'bio_exact', 'bio_single')}


def main(teacher, baseline, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    weights = teacher / 'model.safetensors'
    if sha(weights) != WEIGHTS_SHA:
        raise ValueError('Independent teacher weights do not match pinned revision')
    baseline_sha = sha(baseline)
    source = json.loads(baseline.read_text(encoding='utf-8'))
    groups = source['groups']
    model = AutoModelForTokenClassification.from_pretrained(
        teacher, local_files_only=True, use_safetensors=True, trust_remote_code=False).eval()
    tokenizer = AutoTokenizer.from_pretrained(teacher, local_files_only=True, trust_remote_code=False, use_fast=True)
    rows = [row for group in groups.values() for row in group['rows']]
    for row, predictions in zip(rows, annotate(model, tokenizer, [row['question'] for row in rows]), strict=True):
        row['teacher'] = predictions
    summary = {group: compare(data['rows']) for group, data in groups.items()}
    if sha(baseline) != baseline_sha:
        raise ValueError('Baseline changed during the audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'teacher': TEACHER, 'revision': REVISION, 'weights_sha256': WEIGHTS_SHA,
                   'baseline_sha256': baseline_sha, 'summary': summary, 'groups': groups},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for group, result in summary.items():
        print(group, result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('teacher', 'baseline', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(**vars(parser.parse_args()))
