"""Exercise scoped proposal handles against an isolated, durable SPORA individual."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_dialogue_evidence import store_json
from scripts.deepseek_local_baseline import AutoTokenizer, MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_named_memory_chat import DEFAULT_PACKED, LocalChat
from scripts.spora_pending_replay_data import REPLIES
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.research_pending_offers import PendingOffers


ARMS = {'expected': ('许乔', '林岑'), 'context': ('齐宁', '陶安'), 'reply': ('宁岚', '赵禾')}
RELATIONS = ('集合点', '零食')


def address(chat, name, relation):
    entity = chat.runtime.resolve_confirmed_name(name)['key']
    if entity is None:
        raise ValueError(f'Could not bind reviewed name: {name}')
    return entity, relation


def inspect(chat, subject, revision=None):
    record = chat.runtime.read_structured_relation(*address(chat, *subject), as_of_revision=revision)
    if record is None:
        return None
    decoded = chat.get_structured_codec().decode(record['packet'], record['metadata'])
    return dict(value=chat.tokenizer.decode(decoded['fact']['token_ids'], skip_special_tokens=True),
                revision=record['revision'])


def prepare(chat, subject, value):
    chat.pending_fact = None
    response = chat.remember_fact(*subject, value, pending=True)
    pending = chat.pending_fact
    chat.pending_fact = None
    if pending is None:
        raise ValueError(f'Unable to prepare a staged fact: {response}')
    return pending


def replay(chat, mode, decisions):
    subjects = tuple(zip(ARMS[mode], RELATIONS, strict=True))
    for subject, initial in zip(subjects, ('北门', '桂花糖'), strict=True):
        chat.register(subject[0])
        address(chat, *subject)
        chat.remember_fact(*subject, initial)
    origin = [inspect(chat, subject) for subject in subjects]
    if [row['value'] for row in origin] != ['北门', '桂花糖']:
        raise ValueError('Controlled seed values did not survive encoding')
    book = PendingOffers(chat.runtime)
    result, history = [], []

    def stage(index, text, turn, valid_through):
        return book.propose(prepare(chat, subjects[index], text), turn=turn, valid_through=valid_through)

    def apply(event, handle, index, turn, *, sentence=None, expected, allow=None, expected_status=None):
        # All addresses are supplied by the scenario, independent of the learned reply heads.
        authorized = decisions[sentence][mode] if sentence is not None and allow is None else allow
        before = [inspect(chat, subject) for subject in subjects]
        status, revision = book.confirm(handle, scope=address(chat, *subjects[index]), turn=turn, authorized=authorized)
        after = [inspect(chat, subject) for subject in subjects]
        writes = status == 'committed'
        if before[1 - index] != after[1 - index] or (not writes and before != after):
            raise ValueError('A limited proposal changed an unrelated record')
        if expected_status and status != expected_status:
            raise ValueError(f'Transaction {event} returned {status}, expected {expected_status}')
        if writes:
            if revision != after[index]['revision'] or before[index]['revision'] >= revision:
                raise ValueError('Applied a write without a fresh record revision')
            history.append(dict(subject=list(subjects[index]), **after[index]))
        result.append(dict(event=event, requested_scope=list(subjects[index]), sentence=sentence,
                           authorized=authorized, expected_write=expected, written=writes, status=status,
                           before=before, after=after))
        return status

    a1 = stage(0, '银杏亭', 1, 12)
    b1 = stage(1, '薄荷糖', 2, 12)
    if [inspect(chat, subject) for subject in subjects] != origin:
        raise ValueError('Prepared records must not change current memory')
    apply('ask_details', a1, 0, 3, sentence='info', expected=False)
    apply('unbound_confirmation', None, 0, 4, sentence='ambiguous', expected=False, expected_status='unbound')
    apply('another_scope', b1, 0, 5, sentence='accept_snack', expected=False, expected_status='wrong_scope')
    apply('changing_topic', b1, 1, 6, sentence='other_topic', expected=False)
    wrote = apply('confirm_interleaved_snack', b1, 1, 7, sentence='accept_snack', expected=True)
    if wrote == 'committed':
        apply('duplicate_confirmation', b1, 1, 7, allow=True, expected=False, expected_status='closed')
    a2 = stage(0, '车站', 8, 13)
    # Even an explicit approval must not resurrect the previous handle.
    apply('replaced_proposal', a1, 0, 9, sentence='accept_place', expected=False, expected_status='superseded')
    apply('confirm_replacement', a2, 0, 10, sentence='accept_changed', expected=True)
    a3 = stage(0, '南门', 11, 16)
    chat.remember_fact(*subjects[0], '东门')
    winner = inspect(chat, subjects[0])
    apply('record_revision_changed', a3, 0, 12, allow=True, expected=False, expected_status='revision_changed')
    if inspect(chat, subjects[0]) != winner:
        raise ValueError('Version conflict replaced the winning record')
    b2 = stage(1, '梨糖', 13, 13)
    apply('expired_proposal', b2, 1, 14, allow=True, expected=False, expected_status='expired')
    b3 = stage(1, '梨糖', 15, 24)
    if book.withdraw(b3, scope=address(chat, *subjects[1]), turn=16) != 'withdrawn':
        raise ValueError('The deliberate cancellation did not close the proposal')
    apply('withdrawn_proposal', b3, 1, 17, allow=True, expected=False, expected_status='closed')
    b4 = stage(1, '梨糖', 18, 24)
    book = PendingOffers(chat.runtime)
    apply('different_session', b4, 1, 19, allow=True, expected=False, expected_status='other_session')
    return {'steps': result, 'snapshots': [dict(subject=list(subject), **before)
                                         for subject, before in zip(subjects, origin, strict=True)] + history,
            'current': [dict(subject=list(subject), **inspect(chat, subject)) for subject in subjects]}


def readback(chat, arm):
    historic = [chat.recall_fact(*row['subject'], revision=row['revision']) for row in arm['snapshots']]
    current = [chat.recall_fact(*row['subject']) for row in arm['current']]
    if historic != [row['value'] for row in arm['snapshots']] or current != [row['value'] for row in arm['current']]:
        raise ValueError('The decoder did not generate the stored revision values')
    return dict(historic=historic, current=current)


def main(args):
    torch.set_num_threads(2)
    target = args.directory.resolve()
    db = target / 'isolated.sqlite'
    destination = target / ('reopened.json' if args.reopen else 'transactions.json')
    if destination.exists() or db.exists() != args.reopen:
        raise FileExistsError('The isolated database does not match the requested phase')
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    decoder = load_full_streamed()
    if args.reopen:
        expected = json.loads((target / 'transactions.json').read_text(encoding='utf-8'))
        if sha(DEFAULT_PACKED) != expected['checkpoint_sha256']:
            raise ValueError('The memory checkpoint changed during the trial')
        before = sha(db)
        with load_integrated_durable(db, DEFAULT_PACKED, decoder, MODEL) as runtime:
            chat = LocalChat(runtime, tokenizer, None, None)
            replayed = {mode: readback(chat, arm) for mode, arm in expected['arms'].items()}
        if replayed != expected['answers'] or sha(db) != before:
            raise ValueError('A read-only reopen changed the recorded individual or replies')
        store_json(destination, dict(answers=replayed, database_sha256=before, seconds=perf_counter() - started))
        print('Verified current and historical replies after restart', flush=True)
        return
    comments = target / 'decisions.json'
    decisions = json.loads(comments.read_text(encoding='utf-8'))['outcomes']
    if set(decisions) != set(REPLIES) or any(decisions[key]['expected'] != row[3] for key, row in REPLIES.items()):
        raise ValueError('The measured decisions do not match the replay messages')
    with create_integrated_durable(db, DEFAULT_PACKED, decoder, MODEL, initial_state_factory=fresh_controlled_state) as runtime:
        chat = LocalChat(runtime, tokenizer, None, None)
        arms = {mode: replay(chat, mode, decisions) for mode in ARMS}
        if any(step['expected_write'] != step['written'] for step in arms['expected']['steps']):
            raise ValueError('The reference replay did not make the expected writes')
        answers = {mode: readback(chat, arm) for mode, arm in arms.items()}
    store_json(destination, dict(arms=arms, answers=answers, seconds=perf_counter() - started,
                                 decisions_sha256=sha(comments), checkpoint_sha256=sha(DEFAULT_PACKED)))
    print({mode: [(step['event'], step['status']) for step in arm['steps']] for mode, arm in arms.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--reopen', action='store_true')
    main(parser.parse_args())
