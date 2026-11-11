"""Audit current-value drift and repeat decisions on synthetic dense histories."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_delayed_fact_audit import current, historical, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_provenance_band_selector_pilot import KEYS
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.internal_memory import enable_event_key_index, enable_independent_provenance, observe_sourced, write_features
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-repeat-current-audit-20261112.json'
VALUES = (1, 3, 4, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key6', action='store_true', help='Audit the key used by the full-model counterfactual')
    args = parser.parse_args()
    output = (ROOT / 'runs/deepseek-spora-repeat-current-key6-20261112.json') if args.key6 else OUTPUT
    keys = (6,) if args.key6 else KEYS
    if output.exists():
        raise FileExistsError(output)
    start = time.perf_counter()
    torch.set_num_threads(2)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    rows = []
    for key in keys:
        for target in VALUES:
            state = enable_independent_provenance(model,
                enable_event_key_index(model, fresh_controlled_state(model)))
            steps = []
            expected_current = None
            for event in range(64):
                value = target if event == 62 else (0 if event == 63 else VALUES[event % len(VALUES)])
                read_before = current(model, state, key)
                _, _, feature_old, _, _ = write_features(model, state, key, value, 0, 1.0, event)
                if read_before != feature_old:
                    raise ValueError('Current query and controller features disagree')
                prior = expected_current
                state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
                if decision['current_write']:
                    expected_current = value
                read_after = current(model, state, key)
                steps.append({'event': event, 'input': value, 'expected_before': prior,
                              'read_before': read_before, 'action': decision['action'],
                              'reason': decision['reason'], 'current_write': decision['current_write'],
                              'expected_after': expected_current, 'read_after': read_after})
            final = select_event(model, state, key, 'effective_previous', as_of=63)
            row = {'key': key, 'target': target, 'steps': steps,
                   'target_history_value': historical(model, state, key, 62), 'selection': final,
                   'read_drift_before': sum(s['expected_before'] is not None and s['read_before'] != s['expected_before'] for s in steps),
                   'read_drift_after': sum(s['expected_after'] is not None and s['read_after'] != s['expected_after'] for s in steps),
                   'repeat_against_last_accepted': sum(s['action'] == 'repeat' and s['expected_before'] is not None
                                                       and s['input'] != s['expected_before'] for s in steps)}
            rows.append(row)
        print(json.dumps({'key': key, 'cases': 4}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Source changed')
    summary = {'cases': len(rows), 'events': sum(len(r['steps']) for r in rows),
               'read_drift_before': sum(r['read_drift_before'] for r in rows),
               'read_drift_after': sum(r['read_drift_after'] for r in rows),
               'repeat_against_last_accepted': sum(r['repeat_against_last_accepted'] for r in rows),
               'target_actions': dict(Counter(r['steps'][62]['action'] for r in rows)),
               'target_history_matches': sum(r['target_history_value'] == r['target'] for r in rows),
               'selected_target': sum(r['selection'].get('event_time') == 62 for r in rows)}
    report = {'status': 'dense_current_repeat_development_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - start,
              'limitations': ['Prior accepted value is a synthetic bookkeeping oracle, not external truth.',
                              'Known numeric keys, four deterministic values and one trajectory family.',
                              'CPU audit may not exactly reproduce GPU floating-point decisions.',
                              'Selection target 62 is a test convention; repeat-event semantics remain open.',
                              'Fresh experimental states and independent provenance, no original state writes.']}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
