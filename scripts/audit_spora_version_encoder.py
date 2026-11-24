"""Measure a small Chinese encoder against character features for version routing."""

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_version_roles import (CAL, CAL_NAMES, SEED, TRAIN, TRAIN_NAMES, contrast_examples,
                                              fit, role_predictions)
from scripts.audit_spora_version_timeline import SUBJECTS, TODAY
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.research_version_roles import ABSTAIN, VersionRoleHead, apply_version_plan, clause_spans
from spora.research_version_timeline import effective_order


WEIGHTS = '354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026'
THRESHOLDS = (0., .35, .45, .55, .65, .75, .85, .95, 1.)


def clauses(question):
    return [question[start:end].strip() for start, end in clause_spans(question)]


def ngrams(text):
    return [text[i:i + width] for width in (1, 2) for i in range(len(text) - width + 1)]


def character_features(texts, training):
    vocab = {term: i for i, term in enumerate(sorted({term for text in training for term in ngrams(text)}))}
    vectors = torch.zeros(len(texts), len(vocab))
    for i, text in enumerate(texts):
        for term, count in Counter(ngrams(text)).items():
            if term in vocab:
                vectors[i, vocab[term]] = count
    return F.normalize(vectors)


@torch.inference_mode()
def encoder_features(texts, path):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False).eval()
    batches = []
    for offset in range(0, len(texts), 32):
        batch = tokenizer(texts[offset:offset + 32], padding=True, return_tensors='pt')
        vectors = model(**batch).last_hidden_state[:, 0]
        batches.append(F.normalize(vectors))
    return torch.cat(batches)


def features(questions, vectors, text_index):
    parts = [[text_index[text] for text in clauses(question)] for question in questions]
    width = max(map(len, parts))
    padded = torch.zeros(len(parts), width, vectors.shape[-1])
    mask = torch.zeros(len(parts), width, dtype=torch.bool)
    for i, indices in enumerate(parts):
        padded[i, :len(indices)] = vectors[indices]
        mask[i, :len(indices)] = True
    return padded, mask


def unseen_questions(schedules):
    cues = ('今天办事依照的那条', '建卡开张就填好的那条', '最后一回换页时撤换下来的那条')
    rows = []
    for subject, (_, name, field, _, _) in enumerate(SUBJECTS):
        tail = f'{name}的{field}写的是哪项？'
        slots = effective_order(schedules[subject], as_of=TODAY, known_by=TODAY)
        if len(slots) < 3:
            raise ValueError('The challenge requires three distinct effective versions')
        coordinates = (slots[-1], slots[0], slots[-2])
        for left, right in ((0, 1), (0, 2), (1, 2)):
            for order in (0, 1):
                pair = f'{subject}-{left}-{right}-{order}'
                for seek, drop in ((left, right), (right, left)):
                    parts = (f'我想对照{cues[seek]}', f'{cues[drop]}留待下回再核')
                    text = '，'.join(parts if order == 0 else parts[::-1]) + '。' + tail
                    rows.append({'split': 'unseen', 'pair': pair, 'question': text,
                                 'address': subject, 'legacy': ABSTAIN, 'gold': [subject, coordinates[seek]]})
        for text in (f'{cues[0]}留待下回再核。{tail}',
                     f'{cues[1]}和{cues[2]}暂时都先收起来。{tail}',
                     f'{cues[2]}本来想核对，眼下先别调出来。{tail}'):
            rows.append({'split': 'unseen', 'pair': None, 'question': text,
                         'address': subject, 'legacy': ABSTAIN, 'gold': None})
    return rows


def classify(prediction, threshold):
    plan = prediction['plan']
    return plan if plan is not None and prediction['confidence'] >= threshold else ABSTAIN


def calibrate(predicted, labeled):
    def score(threshold):
        choices = [classify(item, threshold) for item in predicted]
        targets = [row['target'] for row in labeled]
        pairs = list(zip(choices, targets, strict=True))
        return (-sum(choice != ABSTAIN and choice != target for choice, target in pairs),
                sum(choice == target != ABSTAIN for choice, target in pairs), threshold)

    return max(THRESHOLDS, key=score)


def route(row, schedules, decision=None):
    subject = row['address']
    if subject is None:
        return None
    slot, _ = apply_version_plan(row['question'], schedules[subject], TODAY, row['legacy'], decision)
    return None if slot is None else [subject, slot]


def summarize(rows):
    result = {}
    for split in dict.fromkeys(row['split'] for row in rows):
        items = [row for row in rows if row['split'] == split]
        pairs = {}
        for row in items:
            if row['pair'] is not None:
                pairs.setdefault(row['pair'], []).append(row)
        result[split] = {}
        for mode in items[0]['evaluated']:
            predictions = [(row['evaluated'][mode], row['gold']) for row in items]
            result[split][mode] = {
                'total': len(items), 'positive': sum(gold is not None for _, gold in predictions),
                'correct': sum(pick == gold for pick, gold in predictions),
                'positive_correct': sum(pick == gold and gold is not None for pick, gold in predictions),
                'wrong_read': sum(pick is not None and pick != gold for pick, gold in predictions),
                'negative_read': sum(pick is not None and gold is None for pick, gold in predictions),
                'paired_correct': sum(all(row['evaluated'][mode] == row['gold'] for row in group)
                                      for group in pairs.values()), 'pairs': len(pairs),
            }
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError('Use a fresh output file')
    weights = args.encoder / 'model.safetensors'
    protected = (args.menu, args.timeline, weights, *sorted(args.encoder.glob('*.json')))
    digests = {str(path): base.sha(path) for path in protected}
    if digests[str(weights)] != WEIGHTS:
        raise ValueError('Unexpected BGE model digest')
    previous = json.loads(args.menu.read_text(encoding='utf-8'))
    timeline_hash = next((digest for path, digest in previous['sources'].items()
                          if Path(path).resolve() == args.timeline.resolve()), None)
    if timeline_hash != digests[str(args.timeline)]:
        raise ValueError('Timeline differs from the prior menu audit')
    bank = torch.load(args.timeline, map_location='cpu', weights_only=True)
    schedules = [tuple((date.fromisoformat(e), date.fromisoformat(o)) for e, o in bank['revisions'][name])
                 for _, name, _, _, _ in SUBJECTS]
    training, calibration = contrast_examples(TRAIN_NAMES, TRAIN), contrast_examples(CAL_NAMES, CAL)
    rows = [{key: row[key] for key in ('split', 'question', 'pair', 'address', 'legacy', 'gold')}
            for row in previous['rows']] + unseen_questions(schedules)
    questions = [[row['question'] for row in split] for split in (training, calibration, rows)]
    unique_questions = set().union(*map(set, questions))
    if len(unique_questions) != sum(map(len, questions)):
        raise ValueError('Questions overlap across splits or are duplicated')
    text_index = dict.fromkeys(text for split in questions for question in split for text in clauses(question))
    text_index = {text: index for index, text in enumerate(text_index)}
    training_texts = {text for question in questions[0] for text in clauses(question)}
    for row in rows:
        row['evaluated'] = {'baseline': route(row, schedules)}
    for row, prior in zip(rows[:len(previous['rows'])], previous['rows'], strict=True):
        if row['evaluated']['baseline'] != prior['baseline']:
            raise ValueError('Baseline no longer matches the prior audit')
    thresholds = {}
    for name in ('character', 'bge'):
        vectors = (character_features(text_index, training_texts) if name == 'character'
                   else encoder_features(list(text_index), args.encoder))
        train, calibration_features, evaluation = (features(split, vectors, text_index) for split in questions)
        torch.manual_seed(SEED)
        head = VersionRoleHead(vectors.shape[-1])
        fit({'roles': head}, *train, training, steps=180)
        head.eval()
        calibration_predictions = role_predictions(head, *calibration_features)
        threshold = calibrate(calibration_predictions, calibration)
        thresholds[name] = threshold
        calibration_choices = [classify(item, threshold) for item in calibration_predictions]
        print(name, 'threshold', threshold, 'calibration reads', sum(x != ABSTAIN for x in calibration_choices), flush=True)
        for row, prediction in zip(rows, role_predictions(head, *evaluation), strict=True):
            row.setdefault('proposals', {})[name] = prediction
            for mode, cutoff in ((name, 0.), (name + '_calibrated', threshold)):
                row['evaluated'][mode] = route(row, schedules, classify(prediction, cutoff))
    if any(base.sha(Path(path)) != digest for path, digest in digests.items()):
        raise ValueError('A protected input changed')
    summary = summarize(rows)
    store_json(args.output, {'sources': digests, 'thresholds': thresholds, 'summary': summary, 'rows': rows,
                             'seconds': perf_counter() - started})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('menu', 'timeline', 'encoder', 'output'):
        parser.add_argument(name, type=Path)
    main(parser.parse_args())
