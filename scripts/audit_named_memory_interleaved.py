"""Audit interleaved named histories across three independent processes."""

import argparse
import json
from pathlib import Path

import torch

from scripts.audit_qwen_compatible_natural_memory_20261121 import MODEL, WORDS, extract, load_head, sha
from scripts.audit_qwen_durable_entity_reads_20261122 import read_value, write_fact
from scripts.audit_qwen_packed_reply_20261122 import compare_payloads
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from scripts.train_qwen_counterfactual_reply_20261122 import load_reply_head
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.research_temporal_intent import temporal_logits, time_selection
from transformers import AutoTokenizer


NAMES = ('苗绍宁', '小桐', '蒋韵', '阿路', '叶雯安', '贾哲')
KEYS = (3, 14, 26, 37, 49, 58)
ORDER = ((0, 3, 1, 4, 2, 5), (5, 1, 3, 2, 4, 0), (2, 4, 0, 5, 1, 3))
FORMS = (
    '把最新生效的那份找出来，{name}登记的是哪种饮品？',
    '我在找变更前的记录，{name}上一份填的饮品是什么？',
    '按刚换上的单子，{name}填的饮品叫啥？',
)


def value(person, generation):
    return WORDS[(person * 3 + generation * 5) % len(WORDS)]


def append_generation(runtime, generation):
    for person in ORDER[generation]:
        key = runtime.resolve_confirmed_name(NAMES[person])['key']
        if key != KEYS[person]:
            raise ValueError('Enrollment is missing or mapped to the wrong key')
        write_fact(runtime, key, value(person, generation))


def cases(generation):
    for person, name in enumerate(NAMES):
        for mode, pattern in enumerate(FORMS):
            revision = generation - 1 if mode == 1 else generation
            yield name, pattern.format(name=name), KEYS[person], mode, value(person, revision)
    for name in ('苗绍宁越', '小桐言', '冉白'):
        yield name, FORMS[0].format(name=name), None, 0, None


def assess(runtime, tokenizer, temporal, generation, reply=None, recovery=None):
    trial = list(cases(generation))
    questions = [row[1] for row in trial]
    hidden, mask = extract(runtime.decoder.base, tokenizer, questions, batch_size=1,
                           hidden_indices=tuple(temporal['hidden_indices']), last_logit_only=True)
    modes = choose_modes(temporal, hidden, mask, questions)
    if recovery is not None:
        recovered = choose_modes(recovery, hidden, mask, questions)
        modes = [2 if mode == 3 and other == 2 else mode for mode, other in zip(modes, recovered, strict=True)]
    rows = []
    for (name, question, key, mode, expected), predicted in zip(trial, modes, strict=True):
        address = runtime.resolve_confirmed_name(name)['key']
        known = address is not None
        learned = read_value(runtime, tokenizer, question, address, predicted, reply) if known and predicted != 3 else None
        if mode == predicted:
            control = learned
        else:
            control = read_value(runtime, tokenizer, question, address, mode, reply) if known else None
        rows.append({'name': name, 'question': question, 'expected_key': key, 'resolved_key': address,
                     'expected_mode': mode, 'predicted_mode': predicted, 'expected_value': expected,
                     'learned': learned, 'given_time': control,
                     'learned_passed': address == key and (not known or (mode == predicted and matches(learned, expected))),
                     'given_time_passed': address == key and (not known or matches(control, expected))})
    return rows


def choose_modes(head, hidden, mask, questions):
    probabilities = temporal_logits(head, hidden, mask, texts=questions).softmax(-1)
    return time_selection(probabilities, head['threshold'], factorized=head.get('architecture') == 'factorized').tolist()


def matches(record, expected):
    return record is not None and record['value'] == expected and expected in (record['text'] or '')


def main(args):
    torch.set_num_threads(2)
    digest = sha(args.checkpoint)
    temporal, temporal_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', digest)
    sources = {'checkpoint_sha256': digest, 'temporal_sha256': temporal_sha}
    if args.phase == 'reply':
        return audit_reply(args, temporal, sources)
    if args.phase == 'packed':
        return audit_packed(args, temporal, sources)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f'{args.phase}.json'
    if output.exists():
        raise FileExistsError(output)
    database = args.output / 'individual.sqlite'
    prior = None
    if args.phase != 'initial':
        previous = 'initial' if args.phase == 'update' else 'update'
        prior = json.loads((args.output / f'{previous}.json').read_text(encoding='utf-8'))
        if prior['sources'] != sources:
            raise ValueError('Audit sources changed between phases')
    load = create_integrated_durable if args.phase == 'initial' else load_integrated_durable
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    with load(database, args.checkpoint, load_full_streamed(), MODEL) as runtime:
        if args.phase == 'initial':
            runtime.state = fresh_controlled_state(runtime.organism)
            for name, key in zip(NAMES, KEYS, strict=True):
                runtime.enroll_name(key, name)
            append_generation(runtime, 0)
            append_generation(runtime, 1)
            before = None
        else:
            if runtime.store.revision != prior['revision']:
                raise ValueError('Unexpected individual revision')
            before = assess(runtime, tokenizer, temporal, 1 if args.phase == 'update' else 2)
        if args.phase == 'update':
            append_generation(runtime, 2)
        current = assess(runtime, tokenizer, temporal, 1 if args.phase == 'initial' else 2) if args.phase != 'reopen' else before
        revision, cursor = runtime.store.revision, runtime.state.cursor
    if sha(args.checkpoint) != digest:
        raise ValueError('Bootstrap changed')
    report = {'sources': sources, 'revision': revision, 'cursor': cursor, 'rows': current}
    if prior is not None:
        report['exact_reopen'] = before == prior['rows']
    with output.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(args.phase, 'revision', revision, 'learned', sum(row['learned_passed'] for row in current),
          'given time', sum(row['given_time_passed'] for row in current), 'cases', len(current),
          'reopen parity', report.get('exact_reopen'), flush=True)


def audit_reply(args, temporal, sources):
    path = args.output / 'reply.json'
    if path.exists():
        raise FileExistsError(path)
    prior = json.loads((args.output / 'reopen.json').read_text(encoding='utf-8'))
    if prior['sources'] != sources or not prior.get('exact_reopen'):
        raise ValueError('Reply comparison requires the same verified durable replay')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    with load_integrated_durable(args.output / 'individual.sqlite', args.checkpoint,
                                 load_full_streamed(), MODEL) as runtime:
        if runtime.store.revision != prior['revision']:
            raise ValueError('Durable state changed after baseline')
        reply, reply_sha = load_reply_head(args.reply_head, sources['checkpoint_sha256'], 'cuda')
        rows = assess(runtime, tokenizer, temporal, 2, reply)
    if sha(args.checkpoint) != sources['checkpoint_sha256']:
        raise ValueError('Bootstrap changed')
    summary = {field: [sum(row[field] for row in prior['rows']), sum(row[field] for row in rows)]
               for field in ('learned_passed', 'given_time_passed')}
    with path.open('x', encoding='utf-8') as file:
        json.dump({'sources': dict(sources, reply_sha256=reply_sha), 'revision': prior['revision'],
                   'summary_before_after': summary, 'rows': rows}, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print('reply', summary, 'cases', len(rows), flush=True)


def audit_packed(args, temporal, sources):
    recovered = args.recovery_head is not None
    path = args.output / ('recovery.json' if recovered else 'packed.json')
    database = args.output / 'packed-individual.sqlite'
    if path.exists() or (not recovered and database.exists()):
        raise FileExistsError('Use a new packed audit destination')
    expected = json.loads((args.output / 'reply.json').read_text(encoding='utf-8'))
    if any(expected['sources'][key] != value for key, value in sources.items()):
        raise ValueError('Reply comparison must use the audited source decoder')
    checkpoint = args.packed_checkpoint
    compare_payloads(args.checkpoint, checkpoint)
    directory = checkpoint.parent
    manifest = json.loads((directory / 'packed.json').read_text(encoding='utf-8'))
    verified = json.loads((directory / 'verified.json').read_text(encoding='utf-8'))
    digest = sha(checkpoint)
    if (manifest['source_sha256'] != sources['checkpoint_sha256']
            or manifest['reply_sha256'] != expected['sources']['reply_sha256']
            or manifest['packed_sha256'] != digest or verified['packed_sha256'] != digest
            or not verified['reply_exact'] or not verified['ordinary_exact']):
        raise ValueError('Packed decoder does not match the verified reply arm')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    recovery = None
    output_sources = dict(expected['sources'], packed_sha256=digest)
    loader = load_integrated_durable if recovered else create_integrated_durable
    if recovered:
        original = json.loads((args.output / 'packed.json').read_text(encoding='utf-8'))
        if original['sources'] != output_sources or not original['parity']:
            raise ValueError('Recovery comparison requires an unmodified packed baseline')
        recovery, output_sources['recovery_sha256'] = load_head(
            args.recovery_head, 'temporal.pt', 'temporal_sha256', sources['checkpoint_sha256'])
        if recovery['hidden_indices'] != temporal['hidden_indices']:
            raise ValueError('Time selector and recovery head require the same features')
    with loader(database, checkpoint, load_full_streamed(), MODEL) as runtime:
        if not recovered:
            runtime.state = fresh_controlled_state(runtime.organism)
            for name, key in zip(NAMES, KEYS, strict=True):
                runtime.enroll_name(key, name)
            for generation in range(len(ORDER)):
                append_generation(runtime, generation)
        elif runtime.store.revision != original['revision']:
            raise ValueError('Packed individual changed before recovery audit')
        rows = assess(runtime, tokenizer, temporal, 2, recovery=recovery)
        revision = runtime.store.revision
    parity = rows == expected['rows'] and revision == expected['revision']
    with path.open('x', encoding='utf-8') as file:
        report = {'sources': output_sources, 'revision': revision, 'rows': rows}
        if not recovered:
            report['parity'] = parity
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print('packed recovery' if recovered else f'packed parity {parity}', 'learned', sum(row['learned_passed'] for row in rows),
          'given time', sum(row['given_time_passed'] for row in rows), 'cases', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'temporal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--phase', choices=('initial', 'update', 'reopen', 'reply', 'packed'), required=True)
    parser.add_argument('--reply-head', type=Path)
    parser.add_argument('--packed-checkpoint', type=Path)
    parser.add_argument('--recovery-head', type=Path)
    args = parser.parse_args()
    if (args.phase == 'reply') != (args.reply_head is not None):
        parser.error('--reply-head is required only for the reply phase')
    if (args.phase == 'packed') != (args.packed_checkpoint is not None):
        parser.error('--packed-checkpoint is required only for the packed phase')
    if args.recovery_head is not None and args.phase != 'packed':
        parser.error('--recovery-head requires the packed phase')
    main(args)
