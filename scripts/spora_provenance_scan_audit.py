"""Audit neural provenance as an event locator, without using a factual index at inference."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, query
from spora.internal_memory import observe_sourced, read_provenance
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-provenance-scan-audit-v2-20261111.json'


@torch.no_grad()
def audit(model, state, writes, keys):
    rows = []
    started = time.perf_counter()
    for key in keys:
        for event in range(state.cursor):
            truth = writes[event]
            provenance = read_provenance(model, state, key, event)
            accepted = provenance.get('source') is not None and 'reason' not in provenance
            logits = query(model, state, torch.tensor([key]), torch.tensor([1]), torch.tensor([event]))
            value = int(logits.argmax(-1))
            belongs = truth['key'] == key
            rows.append({'key': key, 'event': event, 'belongs': belongs,
                         'expected_action': truth['action'] if belongs else None,
                         'expected_value': truth['value'] if belongs else None,
                         'accepted': accepted, 'provenance': provenance, 'value': value})
    true = [r for r in rows if r['belongs']]
    false = [r for r in rows if not r['belongs']]
    selection = []
    for key in keys:
        observed = [r for r in rows if r['key'] == key and r['accepted']
                    and r['provenance']['action'] in ('new', 'correct')]
        expected = [i for i, w in enumerate(writes) if w['key'] == key and w['action'] in ('new', 'correct')]
        for label, select in (('first', min), ('latest', max)):
            actual_time = select(r['event'] for r in observed) if observed else None
            expected_time = select(expected) if expected else None
            selection.append({'key': key, 'which': label, 'expected': expected_time,
                              'predicted': actual_time, 'correct': actual_time == expected_time})
    return {'cursor': state.cursor, 'pages': len(state.pages), 'seconds': time.perf_counter()-started,
            'own_events': len(true), 'own_accepted': sum(r['accepted'] for r in true),
            'own_action_correct': sum(r['accepted'] and r['provenance']['action'] == r['expected_action'] for r in true),
            'own_value_correct': sum(r['accepted'] and r['value'] == r['expected_value'] for r in true),
            'other_events': len(false), 'false_accepts': sum(r['accepted'] for r in false),
            'selection_correct': sum(r['correct'] for r in selection), 'selection_total': len(selection),
            'selection': selection, 'rows': rows}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected individual mismatch')
    torch.manual_seed(4821)
    torch.set_num_threads(2)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    state = fresh_controlled_state(model)
    # Four target writes include corrections and a repeated value; inference is nonfactual.
    events = [(1, 1, 'direct'), (2, 2, 'direct'), (1, 3, 'direct'),
              (2, 4, 'direct'), (1, 5, 'direct'), (1, 5, 'direct'),
              (2, 6, 'inference'), (1, 7, 'direct')]
    events += [(10 + (i % 30), (i * 7) % model.config.unknown, 'direct') for i in range(58)]
    events += [(2, 8, 'direct'), (1, 9, 'direct')]
    writes, phases = [], {}
    for key, value, source in events:
        if state.cursor == capacity(model, state):
            state = grow(model, state)
        state, decision = observe_sourced(model, state, key, value, source, 1.0)
        writes.append({'key': key, 'value': value, 'source': source, **decision})
        if state.cursor in (8, len(events)):
            phases[str(state.cursor)] = audit(model, state, writes, [1, 2, 3, 10, 11, 39, 40])
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Protected individual changed')
    report = {'status': 'controlled_neural_event_scan_audit', 'individual_sha256': INDIVIDUAL_SHA,
              'writes': writes, 'phases': phases,
              'limitations': ['Finite-symbol memory; no language model or natural-language routing.',
                              'Audit ground truth is for scoring only, not used to locate events.',
                              'One selected individual and one deterministic synthetic trajectory.',
                              'Acceptance requires complete provenance, including numeric fields.',
                              'Correct provenance and value do not establish chronological query semantics.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: {n: v for n, v in p.items() if n not in ('rows', 'selection')} for k, p in phases.items()}))


if __name__ == '__main__':
    run()
