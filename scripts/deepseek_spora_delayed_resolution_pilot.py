"""Explicit confirm/reject experiment on the verified integrated individual."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_delayed_fact_audit import (OUTPUT as PRIOR, SCENARIOS, current, historical,
                                                        load_individual_from_verified_checkpoint)
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, save
from spora.historical_resolutions import HistoricalResolutions
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-delayed-resolution-pilot-v2-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha or prior['summary']['previous_abstained'] != 6:
        raise ValueError('Prior unresolved evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_before = frozen_digest(model, original)
    rows = []
    for person in (0, 3):
        key = ENROLLMENTS[person][0]
        distractor = ENROLLMENTS[3 if person == 0 else 0][0]
        for scenario_index, scenario in enumerate(SCENARIOS):
            for decision in ('confirm', 'reject'):
                model.historical_resolutions = HistoricalResolutions(model.config.key_count, capacity=4)
                state = fresh_controlled_state(model)
                target_events, accepted = [], []
                for event, (subject, value, valid_at) in enumerate(scenario):
                    state, result = __import__('spora.internal_memory', fromlist=['observe_sourced']).observe_sourced(
                        model, state, key if subject == 'T' else distractor, value, 'direct', 1.0, valid_at=valid_at)
                    if subject == 'T':
                        target_events.append((event, value, valid_at))
                        if result['action'] in ('new', 'correct'):
                            accepted.append((event, value, valid_at))
                ordered = sorted(target_events if decision == 'confirm' else accepted, key=lambda e: (e[2], e[0]))
                expected = ordered[-2][0] if len(ordered) >= 2 else None
                late_event, late_value, late_valid = target_events[-1]
                before = select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
                before_current, before_historical = current(model, state, key), historical(model, state, key, late_event)
                model.historical_resolutions.record(model, state, key, late_event, value=late_value,
                                                     valid_at=late_valid, decision=decision)
                after = select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
                actual = after.get('event_time') if after['status'] == 'experimental_candidate' else None
                memory_unchanged = (current(model, state, key) == before_current
                                    and historical(model, state, key, late_event) == before_historical)
                digest = frozen_digest(model, state)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'individual.pt'
                    save(path, model, state, {'purpose': 'explicit-historical-resolution-pilot'})
                    loaded, restored, _ = load(path, 'cpu')
                    restored_after = select_event(loaded, restored, key, 'effective_previous', as_of=restored.cursor - 1)
                    reload_same = (frozen_digest(loaded, restored) == digest and restored_after == after)
                selection_correct = ((after['status'] == 'abstain' and after.get('reason') == 'insufficient_effective_candidates')
                                     if expected is None else (after['status'] == 'experimental_candidate' and actual == expected))
                row = {'person': person, 'scenario': scenario_index, 'decision': decision,
                       'expected_previous': expected, 'before': before, 'after': after,
                       'selection_correct': selection_correct, 'memory_unchanged': memory_unchanged,
                       'reload_same': reload_same}
                rows.append(row)
                print(json.dumps({'person': person, 'scenario': scenario_index, 'decision': decision,
                                  'selection_correct': row['selection_correct'],
                                  'memory_unchanged': memory_unchanged, 'reload_same': reload_same}), flush=True)
    del model.historical_resolutions
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or frozen_digest(model, original) != original_before:
        raise ValueError('Protected checkpoint or original individual changed')
    summary = {}
    for decision in ('confirm', 'reject'):
        subset = [r for r in rows if r['decision'] == decision]
        summary[decision] = {'cases': len(subset), 'pre_abstained': sum(r['before']['status'] == 'abstain' for r in subset),
                             'selection_correct': sum(r['selection_correct'] for r in subset),
                             'memory_unchanged': sum(r['memory_unchanged'] for r in subset),
                             'same_process_reload_same': sum(r['reload_same'] for r in subset)}
    result = {'status': 'explicit_resolution_development_pilot', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Twelve cases reuse six synthetic development trajectories and explicit oracle adjudications.',
                              'Records are exact tensor metadata, not learned semantic memory or automatic truth assessment.',
                              'Selection rechecks neural history and provenance, which can still be inaccurate.',
                              'Same-process temporary save and reload, not crash or concurrent update testing.',
                              'No default runtime UI, identity authorization, or natural-language confirmation implemented.',
                              'Caller must make a private copy and persist via a guarded transaction; record mutates in place.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
