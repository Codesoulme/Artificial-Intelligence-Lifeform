"""Measure name-span continuation with independently annotated Chinese text."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_spora_missing_name_roles import (
    CALIBRATION, CALIBRATION_NAMES, CHALLENGE, CHALLENGE_NAMES, FRESH, FRESH_NAMES, HELD_NAMES, HELD_TEXT, HELD_VALUES,
    MODEL, SEED, TRAIN, TRAIN_NAMES, AutoTokenizer, FrozenTurnFeatures, JointAddress, JointNameRoleHead, NameSpanHead,
    atomic_save, calibrate_recall, choices, examples, frozen_digest, hidden_batch, load_full_streamed,
    load_integrated_runtime, proposals, score_requests, sha, span_metrics, store_json, summarize_requests, torch,
)
from scripts.deepseek_cluener_locator_train import TRAIN as CLUENER_TRAIN, TRAIN_SHA, VALIDATION, VALIDATION_SHA
from scripts.deepseek_cluener_span_audit import REVISION, labeled_names
from scripts.train_spora_open_name_roles import FORMAT as PARENT_FORMAT, fit_proposer, replay_proposer_profiles
from torch.nn import functional as F


FORMAT = 'spora-cluener-name-spans-v1'
CONTINUATION = (6, .001)


def read_catalog(path, banned):
    from pyarrow import parquet

    result = []
    used = set()
    for item in parquet.read_table(path).to_pylist():
        text = item['text']
        if not 8 <= len(text) <= 54 or text in used or any(name in text for name in banned):
            continue
        spans = labeled_names(item)
        if any(not 2 <= b - a <= 4 for a, b in spans):
            continue
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            continue
        used.add(text)
        result.append({'text': text, 'mentions': spans, 'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    return sorted(result, key=lambda row: row['sha256'])


def take_by_count(rows, limits):
    groups = [[row for row in rows if min(2, len(row['mentions'])) == key] for key in range(len(limits))]
    if any(len(group) < limit for group, limit in zip(groups, limits, strict=True)):
        raise ValueError('The disjoint corpus is too small for the specified mixes')
    return [row for group, limit in zip(groups, limits, strict=True) for row in group[:limit]]


def corpus():
    names = TRAIN_NAMES + CALIBRATION_NAMES + FRESH_NAMES + CHALLENGE_NAMES + HELD_NAMES
    held_catalog = read_catalog(VALIDATION, names)
    held = take_by_count(held_catalog, (60, 110, 40))
    held_texts = {row['sha256'] for row in held_catalog}
    held_names = {row['text'][a:b] for row in held for a, b in row['mentions']}
    training = read_catalog(CLUENER_TRAIN, names + tuple(sorted(held_names)))
    selected = take_by_count([row for row in training if row['sha256'] not in held_texts], (96, 256, 128))
    return selected, held


def train_proposers(state, width, featurize, real):
    synthetic = choices(TRAIN_NAMES, TRAIN, '荣安云庆岚薇青岳东梦')
    hidden, lengths = hidden_batch(featurize, synthetic)
    real_hidden, real_lengths = hidden_batch(featurize, real)
    padded_width = max(hidden.shape[1], real_hidden.shape[1])

    def pad(items):
        return F.pad(items, (0, 0, 0, padded_width - items.shape[1]))

    hidden = pad(hidden)
    ids = torch.randperm(len(synthetic), generator=torch.Generator().manual_seed(SEED))[:len(real)]
    other = {'synthetic': ([synthetic[i] for i in ids.tolist()], hidden[ids], lengths[ids]),
             'mixed': (real, pad(real_hidden), real_lengths)}
    candidates = {}
    for label, (rows, added, sizes) in other.items():
        head = NameSpanHead(width)
        head.load_state_dict(state, strict=True)
        print('Continuation', label, len(synthetic) + len(rows), flush=True)
        candidates[label] = fit_proposer(head, torch.cat((hidden, added)), torch.cat((lengths, sizes)),
                                         [*synthetic, *rows], epochs=CONTINUATION[0], lr=CONTINUATION[1])
    return candidates


def review_roles(rows, featurize, configs):
    hidden, lengths = hidden_batch(featurize, rows)
    report = {}
    for name, (proposer, role, floor, margin) in configs.items():
        spans = proposals(proposer, hidden, lengths, floor)
        scores = score_requests(role, rows, hidden, spans)
        reviewed = summarize_requests(rows, spans, scores, margin)
        report[name] = {'spans': span_metrics(rows, spans), **reviewed}
        print(name, report[name]['spans'], reviewed['summary'], flush=True)
    return report


def review_news(rows, featurize, configs):
    hidden, lengths = hidden_batch(featurize, rows)
    report = {}
    for name, (proposer, _, floor, _) in configs.items():
        found = proposals(proposer, hidden, lengths, floor)
        scores = span_metrics(rows, found)
        scores['exact_sentences'] = sum(row['mentions'] == spans for row, spans in zip(rows, found, strict=True))
        scores['false_positives_in_empty'] = sum(bool(spans) and not row['mentions']
                                                 for row, spans in zip(rows, found, strict=True))
        scores['rows'] = [{'sha256': row['sha256'], 'expected': row['mentions'], 'proposed': spans}
                          for row, spans in zip(rows, found, strict=True)]
        report[name] = scores
        print('CLUENER', name, {key: value for key, value in scores.items() if key != 'rows'}, flush=True)
    return report


def main(checkpoint, addresses, prior, output, verify):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    paths = {'checkpoint': checkpoint, 'addresses': addresses, 'prior': prior, 'corpus_train': CLUENER_TRAIN,
             'corpus_validation': VALIDATION}
    hashes = {key: sha(value) for key, value in paths.items()}
    if (hashes['corpus_train'], hashes['corpus_validation']) != (TRAIN_SHA, VALIDATION_SHA):
        raise ValueError('Unexpected CLUENER snapshot')
    base = torch.load(addresses, map_location='cpu', weights_only=True)
    if base['format'] != 'spora-entity-field-address-probe-v1' or base['checkpoint_sha256'] != hashes['checkpoint']:
        raise ValueError('Field addresses do not match the model')
    hashes['individual'] = base['individual_sha256']
    parent = torch.load(prior, weights_only=True, map_location='cpu')
    sources = {name: hashes[name] for name in ('checkpoint', 'addresses', 'individual')}
    if parent['format'] != PARENT_FORMAT or parent['sources'] != sources:
        raise ValueError('The parent role and span weights do not match the model')
    saved = torch.load(verify / 'proposers.pt', weights_only=True, map_location='cpu') if verify else None
    if saved is not None and (saved['format'] != FORMAT or saved['sources'] != hashes):
        raise ValueError('Continuation weights do not match this experiment')
    real, news_test = corpus()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(checkpoint, load_full_streamed(), MODEL, disposable_base=True)
    runtime.decoder.eval().requires_grad_(False)
    if frozen_digest(runtime.organism, runtime.state) != hashes['individual']:
        raise ValueError('The field state belongs to a different individual')
    featurize = FrozenTurnFeatures(runtime.decoder, tokenizer)
    original = NameSpanHead(base['hidden']).eval().requires_grad_(False)
    original.load_state_dict(parent['proposer'], strict=True)
    role = JointNameRoleHead(base['hidden']).eval().requires_grad_(False)
    role.load_state_dict(parent['role'], strict=True)
    if saved is None:
        adapted = train_proposers(parent['proposer'], base['hidden'], featurize, real)
    else:
        adapted = {}
        for name, state in saved['heads'].items():
            head = NameSpanHead(base['hidden']).eval().requires_grad_(False)
            head.load_state_dict(state, strict=True)
            adapted[name] = head
    cal = choices(CALIBRATION_NAMES, CALIBRATION, '嘉河亭琛柠添')
    hidden, lengths = hidden_batch(featurize, cal)
    thresholds = {name: calibrate_recall(head, hidden, lengths, cal)[0] for name, head in
                  {'original': original, **adapted}.items()}
    configs = {'strict': (original, role, 0., 0.), 'wider': (original, role, thresholds['original'], 0.)}
    configs.update({name: (head, role, thresholds[name], 0.) for name, head in adapted.items()})
    print('Calibration floors', thresholds, flush=True)
    news = review_news(news_test, featurize, configs)
    development = review_roles(choices(FRESH_NAMES, FRESH, '飞川宁华星明'), featurize, configs)
    challenge = review_roles(choices(CHALLENGE_NAMES, CHALLENGE, '延竹鸿甫初泉'), featurize, configs)
    joint = JointAddress(base['hidden']).eval().requires_grad_(False)
    joint.load_state_dict(base['heads']['JointAddress'], strict=True)
    held = examples(HELD_NAMES, HELD_TEXT, HELD_VALUES, SEED + 1)
    slots = replay_proposer_profiles(held, featurize, configs, joint, tokenizer)
    for name, result in slots.items():
        print('Slots', name, result['summary'], flush=True)
    if (any(sha(path) != hashes[name] for name, path in paths.items())
            or frozen_digest(runtime.organism, runtime.state) != hashes['individual']):
        raise ValueError('A source or the individual changed during the audit')
    record = {'format': FORMAT, 'sources': hashes, 'revision': REVISION, 'thresholds': thresholds,
              'corpus_training': [row['sha256'] for row in real], 'news': news, 'development': development,
              'challenge': challenge, 'slot_replay': slots}
    output.mkdir(parents=True, exist_ok=False)
    if saved is None:
        atomic_save({'format': FORMAT, 'sources': hashes, 'heads': {name: head.state_dict() for name, head in adapted.items()}},
                    output / 'proposers.pt')
        store_json(output / 'result.json', record)
    else:
        reference = json.loads((verify / 'result.json').read_text(encoding='utf-8'))
        if json.loads(json.dumps(record)) != reference:
            raise ValueError('Reloaded results differ from the original audit')
        store_json(output / 'result.json', {'identical': True, 'sources': hashes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'addresses', 'prior', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', type=Path)
    main(**vars(parser.parse_args()))
