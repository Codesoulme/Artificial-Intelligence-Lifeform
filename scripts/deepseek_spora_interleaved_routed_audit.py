"""Audit the existing language router on an isolated interleaved individual.

Facts are programmed into a fresh in-memory state. Questions alone go through
the actual runtime router; no caller-supplied address or time enters answer().
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_interleaved_native_probe import schedule
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import observe_sourced
from spora.identity_fingerprints import IdentityFingerprints
from spora.identity_read_guard import IdentityReadGuard

OUT = ROOT / 'runs/deepseek-spora-interleaved-routed-audit-20261116.json'


def make_cases():
    cases = []
    for index in range(len(NAMES)):
        cases.extend((
            {'question': f'请问{NAMES[index]}以前喜欢喝什么饮料？',
             'index': index, 'phase': 0, 'mode': 1},
            {'question': f'请问{NAMES[index]}现在喜欢喝什么饮料？',
             'index': index, 'phase': 1, 'mode': 0},
        ))
    for question in ('小雪现在喜欢喝什么饮料？', '阿松以前喜欢喝什么饮料？',
                     '小林林现在喜欢喝什么饮料？', '小林子以前喜欢喝什么饮料？',
                     '不要查询小林的饮料记录。', '请给小林写一句鼓励的话。'):
        cases.append({'question': question, 'index': None, 'phase': None, 'mode': None})
    return cases


def score(case, result, events):
    if case['index'] is None:
        return {'negative_rejected': not result['accepted']}
    event = events[(case['index'], case['phase'])]
    key_correct = result.get('key') == event['key']
    mode_correct = result.get('temporal_intent') == case['mode']
    selection = result.get('memory_event_selection')
    time_correct = (case['phase'] == 1 or
                    (isinstance(selection, dict) and selection.get('event_time') == event['time']))
    answer = result.get('answer')
    answer_exact = isinstance(answer, str) and answer.strip().rstrip('。.!！ ').strip() == event['value']
    return {'key_correct': key_correct, 'mode_correct': mode_correct, 'time_correct': time_correct,
            'route_correct': bool(result['accepted'] and key_correct and mode_correct and time_correct),
            'answer_exact': bool(result['accepted'] and key_correct and mode_correct and time_correct and answer_exact)}


def main(output=OUT, fingerprint_guard=False):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': runtime.MODEL / 'model.safetensors', 'integrated': runtime.CHECKPOINT,
             'router': runtime.ROUTER / 'router.pt', 'entity': runtime.ENTITY / 'gate.pt',
             'intent': runtime.INTENT / 'intent.pt', 'temporal': runtime.TEMPORAL}
    hashes = {name: runtime.sha(path) for name, path in paths.items()}
    with tempfile.TemporaryDirectory(prefix='spora-route-audit-') as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'not-created.pt'
        chat = runtime.load_chat()
        guard = None
        if fingerprint_guard:
            fingerprints = IdentityFingerprints(chat.organism.config.key_count)
            for index, name in enumerate(NAMES):
                fingerprints.bind(KEYS[index], name)
            guard = IdentityReadGuard(fingerprints, chat.tokenizer)
        state = fresh_controlled_state(chat.organism)
        events = {}
        for index, value, phase in schedule(tuple(range(len(NAMES)))):
            before = state.cursor
            state, decision = observe_sourced(chat.organism, state, KEYS[index], value, 'direct', 1.0)
            if not decision['current_write'] or state.cursor != before + 1:
                raise ValueError(f'Controlled write rejected: {decision}')
            events[(index, phase)] = {'name': NAMES[index], 'key': KEYS[index],
                                      'value': WORDS[value], 'time': before, 'action': decision['action']}
        chat.state = state
        rows = []
        for case in make_cases():
            result = chat.answer(case['question'], identity_guard=guard)
            rows.append({'case': case, 'result': result, 'score': score(case, result, events)})
            print(json.dumps({'case': len(rows), 'accepted': result['accepted'],
                              'reason': result['reason']}, ensure_ascii=False), flush=True)
        if runtime.WORKING_CHECKPOINT.exists():
            raise ValueError('Audit unexpectedly created a working individual')
    if any(runtime.sha(path) != hashes[name] for name, path in paths.items()):
        raise ValueError('Protected source changed')
    positive = [row for row in rows if row['case']['index'] is not None]
    negative = [row for row in rows if row['case']['index'] is None]
    summary = {'positive': len(positive), 'route_correct': sum(row['score']['route_correct'] for row in positive),
               'answer_exact_and_route_correct': sum(row['score']['answer_exact'] for row in positive),
               'negative': len(negative), 'negative_rejected': sum(row['score']['negative_rejected'] for row in negative),
               'negative_accepted': sum(not row['score']['negative_rejected'] for row in negative)}
    result = {'summary': summary, 'events': list(events.values()), 'rows': rows, 'source_sha256': hashes,
              'fingerprint_guard': fingerprint_guard,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Facts, values, source and event order were provided directly by the test.',
                              'Existing known names, finite beverage vocabulary and authored question forms.',
                              'Six negatives are development controls, not identity-safety or population metrics.',
                              'Uses persisted integrated output adapter and replacement layers, not the oracle-only probe.',
                              'Identity guard, when enabled, uses manually enrolled names and is not authentication.',
                              'Guard cannot reliably detect truncated names or disambiguate homonyms.',
                              'No autonomous observation, long-form conversation, reload or crash recovery.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fingerprint-guard', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    main(args.output, args.fingerprint_guard)
