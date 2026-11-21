"""Compare latest-version routing and explicit correction anchors on longer histories."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer

from scripts.audit_qwen_compatible_natural_memory_20261121 import (
    KEYS, NAMES, WORDS, answer, entity_query_proposer, load_head,
)
from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, save
from spora.identity_fingerprints import IdentityFingerprints
from spora.integrated_checkpoint import load_integrated_runtime
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced
from spora.language_training import frozen_digest
from spora.research_temporal_intent import temporal_logits, time_selection
from spora.revision_routing import parse_revision_reference, select_numbered_revision
from spora.revision_subject import revision_subject


PATTERNS = (
    ('第一次更正之前，给{name}登记的是哪种饮品？', 0),
    ('第1次更正后，{name}的饮品栏填的是什么？', 1),
    ('请查{name}第2次修订前记录的饮品。', 1),
    ('第2次修订完成后，{name}被记成喝什么？', 2),
    ('{name}在第三次更新以前登记的是哪种饮品？', 2),
    ('第三次更新生效后，给{name}登记的饮品是什么？', 3),
    ('当前有效的记录里，{name}喜欢喝什么饮品？', 3),
    ('第4次更正之后，{name}登记了哪种饮品？', None),
    ('第1次更正前和第3次更正后，{name}的饮品分别是什么？', None),
    ('{name}第2次更正的饮品是什么，前后不用区分？', None),
    ('把「{name}第一次更正前的饮品是什么」译成英文，不用查记录。', None),
)
HELD_PATTERNS = (
    ('麻烦查{name}第1次调整发生前登记的饮品。', 0),
    ('第1次调整生效后，为{name}登记的饮品是哪种？', 1),
    ('想知道{name}在第2次修改以前采用的饮品。', 1),
    ('第二次修改完成后，{name}采用的是哪款饮品？', 2),
    ('核对{name}第三次修订前的饮品资料。', 2),
    ('第三次修订之后，替{name}记录的饮品是什么？', 3),
    ('说说{name}现行记录的饮品。', 3),
    ('核对{name}璇第1次调整发生前登记的饮品。', None),
    ('第2次修改完成后，{name}雯登记的饮品是哪款？', None),
    ('想知道{name}和夏兰在第2次修改以前采用的饮品。', None),
    ('第三次修订前后，{name}有什么两样饮品？', None),
    ('第4次修订之后，替{name}登记的饮品是什么？', None),
    ('请翻成英文：「第2次修改完成后，{name}采用的是哪款饮品？」', None),
)


def cases(patterns=PATTERNS):
    return [(phrase.format(name=name), key if index is not None else None, index)
            for name, key in zip(NAMES, KEYS, strict=True) for phrase, index in patterns]


def fixture(organism, offset, *, cross_page=False):
    state = enable_independent_provenance(organism, enable_event_key_index(organism, fresh_controlled_state(organism)))
    expected = {key: [] for key in KEYS}
    for step in range(4):
        if cross_page and step == 3:
            noise = [key for key in range(organism.config.key_count) if key not in KEYS]
            while state.cursor < capacity(organism, state):
                event = state.cursor
                state, _ = observe_sourced(organism, state, noise[event % len(noise)],
                                          (event + offset) % len(WORDS), 'direct', 1., valid_at=event)
        for index, key in enumerate(KEYS):
            if state.cursor == capacity(organism, state):
                state = grow(organism, state)
            value = (index + offset + step * 3) % len(WORDS)
            position = state.cursor
            state, decision = observe_sourced(organism, state, key, value, 'direct', 1., valid_at=position)
            if decision['action'] != ('new' if not step else 'correct') or not decision['current_write']:
                raise ValueError(f'Fixture write rejected at {position}: {decision}')
            expected[key].append({'value': WORDS[value], 'event': position})
            if step == 1:
                if state.cursor == capacity(organism, state):
                    state = grow(organism, state)
                state, duplicate = observe_sourced(organism, state, key, value, 'direct', 1.)
                if duplicate['action'] not in ('repeat', 'support') or duplicate['current_write']:
                    raise ValueError('A duplicate was counted as a new value')
    return state, expected


def current_or_historical(runtime, tokenizer, text, state, key, mode, anchor):
    event = None
    if anchor is not None:
        if anchor['status'] != 'anchored':
            return None, anchor['reason']
        selected = select_numbered_revision(
            runtime.organism, state, key, anchor['number'], anchor['side'],
        )
        if selected['status'] != 'experimental_candidate':
            return None, selected['reason']
        event = selected['event_time']
        mode = 1 if anchor['side'] == 'before' else 2
    result = answer(runtime.decoder, tokenizer, runtime.organism, text, state, key, mode, event=event)
    return result if result['reason'] == 'generated' else None, result['reason']


@torch.no_grad()
def audit(runtime, tokenizer, router, temporal, states, expected, *, questions=None):
    recover_subject = questions is not None
    if questions is None:
        questions = cases()
    texts = [row[0] for row in questions]
    fingerprints = IdentityFingerprints(runtime.organism.config.key_count)
    for name, key in zip(NAMES, KEYS, strict=True):
        fingerprints.bind(key, name)
    propose = entity_query_proposer(runtime.decoder, tokenizer, fingerprints, router, texts)
    features, mask = extract(runtime.decoder.base, tokenizer, texts, batch_size=1,
                             hidden_indices=tuple(temporal['hidden_indices']))
    modes = time_selection(temporal_logits(temporal, features, mask, texts=texts).softmax(-1),
                           temporal['threshold'], factorized=True).tolist()
    variants = ('indexed_ordinal', 'bounded_subject') if recover_subject else ('latest_only', 'indexed_ordinal')
    rows = []
    for i, ((text, expected_key, version), mode) in enumerate(zip(questions, modes, strict=True)):
        chosen, _, address_reason, selected_name = propose(i, text, None)
        anchor = parse_revision_reference(text)
        row = {'question': text, 'expected_key': expected_key, 'expected_version': version,
               'selected_key': chosen, 'selected_name': selected_name, 'mode': mode, 'anchor': anchor}
        recovery = None
        if recover_subject and anchor is not None and anchor['status'] == 'anchored':
            recovery = revision_subject(fingerprints, text)
        recent = {}
        for variant in variants:
            reference = None if variant == 'latest_only' else anchor
            key = chosen
            reason = address_reason
            if variant == 'bounded_subject' and recovery is not None:
                if recovery['key'] is not None:
                    if key is None:
                        key, reason = recovery['key'], None
                    elif key != recovery['key']:
                        key, reason = None, 'revision_subject_disagreement'
                elif recovery['reason'] in ('explicit_name_not_enrolled', 'ambiguous_revision_subject'):
                    key, reason = None, recovery['reason']
            answers, reasons = [], []
            for state_index, state in enumerate(states):
                if key is None or mode == 3:
                    result, attempt_reason = None, reason if key is None else 'read_intent_abstained'
                else:
                    cache_key = state_index, key, reference is not None
                    if cache_key not in recent:
                        recent[cache_key] = current_or_historical(runtime, tokenizer, text, state, key, mode, reference)
                    result, attempt_reason = recent[cache_key]
                answers.append(result)
                reasons.append(attempt_reason)
            event_pair = False
            passed = expected_key is None and all(result is None for result in answers)
            if expected_key is not None and key == expected_key and all(result is not None for result in answers):
                expected_values = [mapping[expected_key][version] for mapping in expected]
                events = [oracle['event'] if anchor is not None else None for oracle in expected_values]
                event_pair = all(result['event_time'] == event for result, event in zip(answers, events, strict=True))
                passed = event_pair and all(
                    result['matrix_value'] == oracle['value']
                    and result['answer'].rstrip('。！ ') == oracle['value']
                    for result, oracle in zip(answers, expected_values, strict=True)
                )
            row[variant] = {'answers': answers, 'reasons': reasons, 'correct_events': event_pair, 'passed': passed}
            if recover_subject:
                row[variant]['key'] = key
        if recover_subject:
            row['subject_recovery'] = recovery
        if expected_key is not None and anchor is not None:
            row['oracle_key_events'] = all(
                select_numbered_revision(runtime.organism, state, expected_key, anchor['number'], anchor['side']).get(
                    'event_time') == oracle[expected_key][version]['event']
                for state, oracle in zip(states, expected, strict=True)
            )
        rows.append(row)
    positives = [r for r in rows if r['expected_key'] is not None]
    negatives = [r for r in rows if r['expected_key'] is None]
    anchored = [r for r in positives if r['anchor'] is not None]
    summary = {'known': len(positives), 'anchored': len(anchored), 'unsupported': len(negatives),
               'correct_name': sum(r['selected_key'] == r['expected_key'] for r in positives),
               'oracle_key_events': sum(r['oracle_key_events'] for r in anchored)}
    for name in variants:
        summary[name] = {'answers': sum(row[name]['passed'] for row in positives),
                         'anchored_answers': sum(row[name]['passed'] for row in anchored),
                         'anchored_events': sum(row[name]['correct_events'] for row in anchored),
                         'unsupported_read': sum(not row[name]['passed'] for row in negatives)}
        if recover_subject:
            summary[name]['correct_name'] = sum(row[name]['key'] == row['expected_key'] for row in positives)
    return {'summary': summary, 'rows': rows}


def main(args):
    output = args.output
    result_path = output / ('reloaded.json' if args.reopen else 'result.json')
    if result_path.exists() or (not args.reopen and output.exists()):
        raise FileExistsError(result_path)
    started = time.perf_counter()
    torch.set_num_threads(2)
    source_sha = sha(args.checkpoint)
    router_source = sha(args.route_source)
    router, router_sha = load_head(args.router, 'router.pt', 'router_sha256', router_source)
    temporal, temporal_sha = load_head(args.temporal, 'temporal.pt', 'temporal_sha256', source_sha)
    if router['hidden_indices'] != temporal['hidden_indices'] or 'past_override' not in temporal['version_residual']:
        raise ValueError('The numbered-history candidate needs the verified identity and temporal heads')
    fingerprints = {'checkpoint_sha256': source_sha, 'router_sha256': router_sha, 'temporal_sha256': temporal_sha}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    runtime = load_integrated_runtime(args.checkpoint, load_full_streamed(), MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    if args.reopen:
        baseline = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        if fingerprints != baseline['sources'] or baseline.get('extended', False) != args.extended:
            raise ValueError('Checkpoint or classifiers changed')
        expected = [{int(key): history for key, history in phase.items()} for phase in baseline['fixture']]
    else:
        output.mkdir(parents=True, exist_ok=False)
        expected = []
        for offset in (0, 2):
            state, values = fixture(runtime.organism, offset, cross_page=args.extended)
            save(output / f'state-{offset}.pt', runtime.organism, state)
            expected.append(values)
    states = []
    digests = []
    for offset in (0, 2):
        restored, state, _ = load(output / f'state-{offset}.pt', 'cpu')
        source_digest = frozen_digest(restored, state)
        if frozen_digest(runtime.organism, state) != source_digest:
            raise ValueError('Restored history does not belong to the loaded runtime')
        digests.append(source_digest)
        states.append(state)
    del restored
    gc.collect()
    torch.cuda.empty_cache()
    if args.extended:
        suites = {}
        for label, questions in (('original', cases()), ('new', cases(HELD_PATTERNS))):
            suites[label] = audit(runtime, tokenizer, router, temporal, states, expected, questions=questions)
            print(label, suites[label]['summary'], flush=True)
        report = {'extended': True, 'suites': suites}
    else:
        report = audit(runtime, tokenizer, router, temporal, states, expected)
    if (frozen_digest(runtime.organism, runtime.state) != individual
            or [frozen_digest(runtime.organism, state) for state in states] != digests
            or sha(args.checkpoint) != source_sha):
        raise ValueError('Numbered-memory query changed a protected model or state')
    if args.reopen:
        root = 'suites' if args.extended else 'rows'
        if digests != baseline['state_sha256'] or report[root] != baseline[root]:
            raise ValueError('Independent reload did not reproduce the numbered-history evaluation')
        report['exact_reopen'] = True
    report.update({'sources': fingerprints, 'state_sha256': digests, 'fixture': expected,
                   'observations_per_state': [state.cursor for state in states],
                   'pages_per_state': [len(state.pages) for state in states],
                   'seconds': time.perf_counter() - started})
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    if not args.extended:
        print(report['summary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'route-source', 'router', 'temporal', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reopen', action='store_true')
    parser.add_argument('--extended', action='store_true', help='Compare subject recovery on two-page histories')
    main(parser.parse_args())
