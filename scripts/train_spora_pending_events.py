"""Audit context-conditioned approval with an explicitly inferred pending state."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_crosswoz_revisions import corpus_rows, validation
from scripts.audit_spora_dialogue_evidence import load_spora_for_eval, store_json
from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.deepseek_local_baseline import sha, torch
from scripts.spora_crosswoz_offer_annotations import TRAIN_ACCEPT, TRAIN_NO_ACCEPT, VALIDATION_QUESTIONS
from scripts.spora_offer_reference_data import REAL
from scripts.spora_pending_event_data import (HELD_CONTRAST, HELD_PIVOT_OPEN, TRAIN_CONTRAST, TRAIN_FOLLOWUP,
                                             TRAIN_PIVOT_CLOSED, TRAIN_PIVOT_OPEN)
from scripts.train_spora_crosswoz_offer_gate import redact, train_archive
from scripts.train_spora_offer_clauses import capture, make_head
from spora.language_training import atomic_save
from torch import nn
from torch.nn import functional as F


SEED = 20261201
EPOCHS = 32
HOLD, APPROVE, PIVOT = range(3)


def episode(dialogue, turn, context, pending, action, family):
    if [item['role'] for item in context] != ['usr', 'sys', 'usr']:
        raise ValueError('The episode needs three ordered speaker turns')
    passages = tuple(redact(item['content'], context) for item in context)
    return dict(key=f'{dialogue}/{turn}', passages=passages, pending=pending, action=action, family=family)


def reviewed(corpus, coordinates, pending, action, family):
    return [episode(dialogue, turn, corpus[dialogue]['messages'][turn:turn + 3], pending, action, family)
            for dialogue, turn in coordinates]


def contrasts(forms, group):
    rows = []
    for number, (offer, answer, user) in enumerate(forms):
        for state, assistant in ((True, offer), (False, answer)):
            rows.append(dict(key=f'{group}/{number}/{int(state)}', family='matched_context',
                             passages=('能说一下目前的安排吗？', assistant + '。', user + '。'),
                             pending=state, action=APPROVE if state else HOLD, pair=f'{group}/{number}'))
    return rows


def dataset(training_path, held_path):
    train_data = train_archive(training_path)
    fit = reviewed(train_data, TRAIN_ACCEPT, True, APPROVE, 'accept')
    fit += reviewed(train_data, TRAIN_NO_ACCEPT, True, HOLD, 'ask_with_offer')
    fit += reviewed(train_data, TRAIN_FOLLOWUP, False, HOLD, 'ask_after_answer')
    fit += reviewed(train_data, TRAIN_PIVOT_CLOSED, False, PIVOT, 'switch_after_answer')
    fit += reviewed(train_data, TRAIN_PIVOT_OPEN, True, PIVOT, 'switch_with_offer')
    del train_data
    held_data = validation(held_path)
    held = []
    for row in corpus_rows(held_data):
        if row['group'] in ('accepted', 'information_only'):
            choice = row['group'] == 'accepted'
            held.extend(reviewed(held_data, [(row['dialogue'], row['turns'][0])], True,
                                 APPROVE if choice else HOLD, 'accept' if choice else 'ask_with_offer'))
    held += reviewed(held_data, VALIDATION_QUESTIONS, True, HOLD, 'ask_with_offer')
    held += reviewed(held_data, HELD_PIVOT_OPEN, True, PIVOT, 'switch_with_offer')
    for dialogue, turn, referent in REAL:
        held.extend(reviewed(held_data, [(dialogue, turn + 1)], False, PIVOT if referent == 2 else HOLD,
                             'switch_after_answer' if referent == 2 else 'ask_after_answer'))
    if {r['key'].split('/')[0] for r in fit}.intersection(r['key'].split('/')[0] for r in held):
        raise ValueError('Training and held conversations overlap')
    if len({r['key'] for r in fit}) != len(fit) or len({r['key'] for r in held}) != len(held):
        raise ValueError('A dialogue transition occurs twice')
    return fit + contrasts(TRAIN_CONTRAST, 'train'), held + contrasts(HELD_CONTRAST, 'held')


def layouts(rows, context):
    for row in rows:
        first, previous, current = row['passages']
        prefix = f'用户：{first}\n助手：'
        if context:
            offset = len(prefix) + len(previous)
            spoken = '\n用户：'
            yield prefix + previous + spoken + current, slice(offset + len(spoken), None), slice(len(prefix), offset)
        else:
            yield '用户：' + current, slice(len('用户：'), None), None


def features(extractor, rows, context):
    samples = list(layouts(rows, context))
    user, offer = [], []
    for start in range(0, len(samples), 12):
        group = samples[start:start + 12]
        values = extractor([row[0] for row in group]).float()
        for vector, (text, user_span, assistant_span) in zip(values, group, strict=True):
            take = vector[:len(text)]
            decision = take[user_span]
            user.append(torch.cat((decision.mean(0), decision[-1])))
            if assistant_span is not None:
                proposal = take[assistant_span]
                offer.append(torch.cat((proposal.mean(0), proposal[-1])))
    return torch.stack(user), torch.stack(offer) if context else None


def classifier(width, count):
    return nn.Sequential(nn.LayerNorm(width, elementwise_affine=False), nn.Linear(width, 48), nn.GELU(), nn.Linear(48, count))


def train(head, values, labels):
    counts = labels.bincount().float()
    balance = len(labels) / (counts * len(counts))
    rng = torch.Generator().manual_seed(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.02)
    trace = []
    for _ in range(EPOCHS):
        total = 0.
        batches = torch.randperm(len(labels), generator=rng).split(16)
        for batch in batches:
            loss = F.cross_entropy(head(values[batch]), labels[batch], weight=balance)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        trace.append(total / len(batches))
    return trace


@torch.no_grad()
def evaluate(heads, evaluated, rows, baseline):
    decisions = {key: model(evaluated[key]).argmax(1).tolist() for key, model in heads.items()}
    summary, detail, pairs = {}, [], {}
    for index, row in enumerate(rows):
        wanted = row['action'] == APPROVE
        previous = bool(decisions['state'][index])
        approval = decisions['context'][index] == APPROVE
        result = {'prior_reply': baseline[index], 'learned_reply': decisions['reply'][index] == APPROVE,
                  'context_only': approval, 'inferred_pending': previous and approval,
                  'given_pending': row['pending'] and approval}
        states = summary.setdefault(row['family'], dict(state_correct=0, action_correct=0, cases=0, modes={}))
        states['cases'] += 1
        states['state_correct'] += previous == row['pending']
        states['action_correct'] += decisions['context'][index] == row['action']
        for mode, active in result.items():
            states['modes'].setdefault(mode, Counter()).update(correct=int(active == wanted),
                                                               false_approval=int(active and not wanted),
                                                               missed_approval=int(wanted and not active))
        detail.append(dict(key=row['key'], family=row['family'], pending=row['pending'], predicted_pending=previous,
                           action=row['action'], predicted_action=decisions['context'][index], authorization=result))
        if row.get('pair'):
            pairs.setdefault(row['pair'], []).append({mode: active == wanted for mode, active in result.items()})
    complete = {mode: sum(len(pair) == 2 and all(item[mode] for item in pair) for pair in pairs.values())
                for mode in result}
    return {'summary': summary, 'rows': detail, 'matched_pairs': complete}


def replay_reply_with_state(evaluation, rows):
    summary, pairs = {}, {}
    for record, row in zip(evaluation['rows'], rows, strict=True):
        if record['key'] != row['key']:
            raise ValueError('The replay row does not match the measured episode')
        active = record['predicted_pending'] and record['authorization']['learned_reply']
        expected = row['action'] == APPROVE
        summary.setdefault(row['family'], Counter()).update(cases=1, correct=int(active == expected),
                                                             false_approval=int(active and not expected),
                                                             missed_approval=int(expected and not active))
        if row.get('pair'):
            pairs.setdefault(row['pair'], []).append(active == expected)
    return {'summary': summary, 'matched_pairs': sum(len(pair) == 2 and all(pair) for pair in pairs.values())}


def main(args):
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=True)
    destination = target / ('reload.json' if args.verify else 'result.json')
    weights, cached = target / 'heads.pt', target / 'features.pt'
    torch.set_num_threads(2)
    if destination.exists() or (not args.verify and any(path.exists() for path in (weights, cached))):
        raise FileExistsError('Use a fresh experiment directory')
    if args.verify:
        original = json.loads((target / 'result.json').read_text(encoding='utf-8'))
        if original['weights_sha256'] != sha(weights) or original['features_sha256'] != sha(cached):
            raise ValueError('Pending-event evaluation inputs changed')
        saved = torch.load(cached, map_location='cpu', weights_only=True)
        states = torch.load(weights, map_location='cpu', weights_only=True)
        heads = {name: classifier(value.shape[-1], 2 if name == 'state' else 3).eval()
                 for name, value in saved['evaluated'].items()}
        for name, head in heads.items():
            head.load_state_dict(states[name])
        if evaluate(heads, **saved) != original['evaluation']:
            raise ValueError('Reopened pending-event heads changed the decisions')
        store_json(destination, dict(verified=len(saved['rows'])))
        print('Verified', len(saved['rows']), 'episodes', flush=True)
        return

    started = perf_counter()
    training, held = dataset(args.training, args.validation)
    print('Dialogue phases', len(training), 'training /', len(held), 'held', flush=True)
    torch.cuda.reset_peak_memory_stats()
    metadata = json.loads((args.prior.parent / 'result.json').read_text(encoding='utf-8'))
    if sha(args.prior) != metadata['sha256'][0]:
        raise ValueError('Expected the verified previous decision head')
    previous = torch.load(args.prior, map_location='cpu', weights_only=True)['supervised']
    chat = load_spora_for_eval(target)
    if {k: v for k, v in chat.awake_individual.items() if k != 'memory_path'} != {
            k: v for k, v in metadata['runtime'].items() if k != 'memory_path'}:
        raise ValueError('Expected the same dialogue backbone used by the previous head')
    chat.tokenizer.padding_side = 'right'
    extractor = FrozenTurnFeatures(chat.decoder, chat.tokenizer)
    fit, evaluated = {}, {}
    for rows, store in ((training, fit), (held, evaluated)):
        store['context'], store['state'] = features(extractor, rows, True)
        store['reply'], _ = features(extractor, rows, False)
    prior_features = capture(extractor, [dict(speaker='usr', reply=row['passages'][2]) for row in held])
    prior = make_head('supervised', prior_features).eval()
    prior.load_state_dict(previous['weights'])
    with torch.no_grad():
        prior_scores, _ = prior(prior_features['clauses'], prior_features['mask'])
    baseline = (prior_scores >= previous['threshold']).tolist()
    heads, loss = {}, {}
    for name, value in fit.items():
        torch.manual_seed(SEED)
        head = classifier(value.shape[-1], 2 if name == 'state' else 3)
        labels = torch.tensor([int(row['pending']) if name == 'state' else row['action'] for row in training])
        loss[name] = train(head, value, labels)
        heads[name] = head.eval()
    data = dict(evaluated=evaluated, rows=held, baseline=baseline)
    metrics = evaluate(heads, **data)
    atomic_save({key: model.state_dict() for key, model in heads.items()}, weights)
    atomic_save(data, cached)
    store_json(destination, {'evaluation': metrics, 'training': len(training), 'runtime': chat.awake_individual,
                             'reply_state_replay': replay_reply_with_state(metrics, held),
                             'loss': loss, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated(),
                             'weights_sha256': sha(weights), 'features_sha256': sha(cached), 'parent_sha256': sha(args.prior)})
    print(metrics['summary'], metrics['matched_pairs'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'validation', 'prior', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
