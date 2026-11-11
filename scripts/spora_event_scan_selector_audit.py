"""Audit an experimental event selector on a multi-page finite-symbol trajectory."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, query, save
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT/'runs/spora-event-scan-selector-audit-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected individual mismatch')
    torch.set_num_threads(2)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    state = fresh_controlled_state(model)
    writes = []
    def write(key, value, source='direct', valid_at=None):
        nonlocal state
        if state.cursor == capacity(model, state):
            state = grow(model, state)
        state, decision = observe_sourced(model, state, key, value, source, 1.0, valid_at=valid_at)
        writes.append({'key': key, 'value': value, 'source': source, 'valid_at': valid_at, **decision})
    for key, value in ((1, 1), (2, 2), (1, 6), (1, 6), (2, 3)):
        write(key, value)
    while state.cursor < 64:
        i = state.cursor
        write(10+i % 30, i % model.config.unknown)
    write(2, 4)
    write(1, 5)
    write(1, 7, valid_at=0)
    write(1, 8, source='inference')
    rows = []
    for key in (1, 2, 3, 10, 39):
        expected_times = [i for i, w in enumerate(writes)
                          if w['key'] == key and w['action'] in ('new', 'correct')]
        for order in ('first', 'previous', 'latest'):
            result = select_event(model, state, key, order)
            index = {'first': 0, 'previous': -2, 'latest': -1}[order]
            expected = expected_times[index] if len(expected_times) >= (2 if order == 'previous' else 1) else None
            actual = result.get('event_time') if result['status'] == 'experimental_candidate' else None
            value = None
            if actual is not None:
                k, mode, t = [torch.tensor([v]) for v in (key, 1, actual)]
                value = int(query(model, state, k, mode, t).argmax(-1))
            rows.append({'key': key, 'order': order, 'expected_event': expected,
                         'result': result, 'native_value': value,
                         'event_correct': actual == expected,
                         'value_correct': None if expected is None else value == writes[expected]['value']})
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'individual.pt'
        save(path, model, state)
        model2, state2, _ = load(path, 'cpu')
        reload_mismatches = [i for i, r in enumerate(rows)
                             if select_event(model2, state2, r['key'], r['order']) != r['result']]
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Protected individual changed')
    report = {'status': 'experimental_finite_symbol_event_scan', 'individual_sha256': INDIVIDUAL_SHA,
              'cursor': state.cursor, 'page_count': len(state.pages), 'writes': writes, 'rows': rows,
              'correct_event_count': sum(r['event_correct'] for r in rows),
              'correct_value_count': sum(r['value_correct'] is True for r in rows),
              'value_cases': sum(r['value_correct'] is not None for r in rows),
              'reload_mismatch_indices': reload_mismatches,
              'limitations': ['Oracle uses controller actions, not independent factual truth.',
                              'Observation order only; valid_at is not used to rank or select.',
                              'Unknown neural reads may conceal events; selection cannot certify completeness.',
                              'Finite symbols, one trajectory, same-process reload; no language model involved.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('cursor', 'page_count', 'correct_event_count',
                                             'correct_value_count', 'value_cases', 'reload_mismatch_indices')}))


if __name__ == '__main__':
    run()
