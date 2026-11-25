"""Evaluate frozen agreement, then exercise it against isolated versioned memory."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_pending_transactions import address, inspect, prepare
from scripts.audit_spora_crosswoz_revisions import validation
from scripts.deepseek_local_baseline import AutoTokenizer, MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_named_memory_chat import DEFAULT_PACKED, LocalChat
from scripts.spora_pending_independent_data import REAL_ACCEPT, REAL_ASK, REAL_CONTINUE, TRANSACTION_KEYS, scripted
from scripts.spora_unknown_state_probe import fresh_controlled_state
from scripts.train_spora_pending_events import APPROVE, HOLD, classifier, features, reviewed
from spora.integrated_durable import create_integrated_durable
from spora.research_pending_offers import PendingOffers


def held_cases(archive, source):
    corpus = validation(archive)
    rows = reviewed(corpus, REAL_ACCEPT, True, APPROVE, 'real_accept')
    rows += reviewed(corpus, REAL_ASK, True, HOLD, 'real_ask')
    rows += reviewed(corpus, REAL_CONTINUE, False, HOLD, 'real_continue')
    reference = torch.load(source / 'features.pt', map_location='cpu', weights_only=True)
    old = {row['key'].split('/')[0] for row in reference['rows']}
    seen = {row['key'].split('/')[0] for row in rows}
    if seen & old or len(seen) != len(rows):
        raise ValueError('An independent dialogue overlaps the previous gate study')
    for row in rows:
        intent = row.pop('action') == APPROVE
        row.update(spoken=intent, expected=intent, binding='open' if row['pending'] else 'unbound')
    return rows + list(scripted())


@torch.no_grad()
def score_cases(chat, rows, weights):
    chat.tokenizer.padding_side = 'right'
    encoder = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    context, state = features(encoder, rows, True)
    reply, _ = features(encoder, rows, False)
    decisions = {}
    for name, values in (('state', state), ('context', context), ('reply', reply)):
        head = classifier(values.shape[-1], 2 if name == 'state' else 3).eval()
        head.load_state_dict(weights[name])
        decisions[name] = head(values).argmax(1).tolist()
    details = []
    for row, proposed, whole, current in zip(rows, decisions['state'], decisions['context'], decisions['reply'], strict=True):
        pending = bool(proposed)
        context = pending and whole == APPROVE
        short = pending and current == APPROVE
        details.append(dict(key=row['key'], family=row['family'], binding=row['binding'], pending=pending,
                            actual_pending=row['pending'], spoken=row['spoken'], expected=row['expected'],
                            pair=row.get('pair'), context=context, reply=short, agreement=context and short))
    return details


def summarize(details):
    metrics, pairs = defaultdict(dict), defaultdict(list)
    for row in details:
        for method in ('context', 'reply', 'agreement'):
            prediction = row[method] and row['binding'] == 'open'
            expected = row['expected']
            stats = metrics[row['family']].setdefault(method, Counter())
            stats.update(cases=1, correct=int(prediction == expected), false_write=int(prediction and not expected),
                         missed_write=int(expected and not prediction), raw_approval=int(row[method]))
        if row['pair']:
            pairs[row['pair']].append(row)
    matched = {method: sum(len(pair) == 2 and all(r[method] == r['expected'] for r in pair)
                          for pair in pairs.values()) for method in ('context', 'reply', 'agreement')}
    return {'by_family': metrics, 'matched_pairs': matched}


def score(args, output):
    started = perf_counter()
    metadata = json.loads((args.weights.parent / 'result.json').read_text(encoding='utf-8'))
    if (sha(args.weights) != metadata['weights_sha256'] or
            sha(args.weights.parent / 'features.pt') != metadata['features_sha256']):
        raise ValueError('Expected the verified pending-event study')
    rows = held_cases(args.validation, args.weights.parent)
    print('Independent held cases:', len(rows), flush=True)
    chat = load_spora_for_eval(output.parent)
    if {k: v for k, v in chat.awake_individual.items() if k != 'memory_path'} != {
            k: v for k, v in metadata['runtime'].items() if k != 'memory_path'}:
        raise ValueError('Unexpected dialogue backbone for the saved heads')
    details = score_cases(chat, rows, torch.load(args.weights, map_location='cpu', weights_only=True))
    summary = summarize(details)
    store_json(output, {'details': details, 'summary': summary, 'weights_sha256': sha(args.weights),
                        'runtime': chat.awake_individual, 'seconds': perf_counter() - started})
    print(summary, flush=True)


def apply(directory, weights, output):
    started = perf_counter()
    scores = directory / 'scores.json'
    scored = json.loads(scores.read_text(encoding='utf-8'))
    if sha(weights) != scored['weights_sha256']:
        raise ValueError('The staged score file uses different authorization weights')
    details = {row['key']: row for row in scored['details']}
    texts = {row['key']: row for row in scripted()}
    db = directory / 'held.sqlite'
    if db.exists():
        raise FileExistsError(db)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    name = '许乔'
    results = []
    with create_integrated_durable(db, DEFAULT_PACKED, base, MODEL, initial_state_factory=fresh_controlled_state) as runtime:
        chat = LocalChat(runtime, tokenizer, None, None)
        chat.register(name)
        for index, key in enumerate(TRANSACTION_KEYS, 1):
            decision, case = details[key], texts[key]
            if any(decision[k] != case[k] for k in ('binding', 'expected')):
                raise ValueError('The replay scope differs from the held manifest')
            subject = (name, f'集合核对{index}')
            chat.remember_fact(*subject, '北门')
            origin = inspect(chat, subject)
            book = PendingOffers(runtime)
            handle = book.propose(prepare(chat, subject, '东门'), turn=1,
                                  valid_through=1 if case['binding'] == 'expired' else 4)
            if case['binding'] == 'superseded':
                book.propose(prepare(chat, subject, '西门'), turn=2, valid_through=4)
            if case['binding'] == 'unbound':
                book.propose(prepare(chat, (name, '点心方案'), '梨糖'), turn=2, valid_through=4)
                handle = None
            if inspect(chat, subject) != origin:
                raise ValueError('A pending record changed before authorization')
            status, revision = book.confirm(handle, scope=address(chat, *subject), turn=3,
                                            authorized=decision['agreement'])
            after = inspect(chat, subject)
            if case['binding'] != 'open' and status != case['binding']:
                raise ValueError('A dead proposal reached the durable write path')
            if (status == 'committed' and after != {'value': '东门', 'revision': revision}
                    or status != 'committed' and after != origin):
                raise ValueError('Unexpected durable transition')
            current = chat.recall_fact(*subject)
            historical = chat.recall_fact(*subject, revision=origin['revision'])
            if current != after['value'] or historical != '北门':
                raise ValueError('The decoder did not read back current and initial facts')
            results.append(dict(key=key, status=status, expected_write=case['expected'],
                                written=status == 'committed', current=current, historical=historical))
    store_json(output, {'rows': results, 'scores_sha256': sha(scores), 'checkpoint_sha256': sha(DEFAULT_PACKED),
                        'seconds': perf_counter() - started})
    print([(row['key'], row['status']) for row in results], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('validation', 'weights', 'directory'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.directory.mkdir(parents=True, exist_ok=True)
    output = args.directory / ('applied.json' if args.apply else 'scores.json')
    if output.exists():
        raise FileExistsError(output)
    if args.apply:
        apply(args.directory, args.weights, output)
    else:
        score(args, output)
