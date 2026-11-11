"""Test explicitly confirmed delayed events under additional memory writes."""
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
from scripts.deepseek_spora_delayed_fact_audit import (SCENARIOS, current, historical,
                                                        load_individual_from_verified_checkpoint)
from scripts.deepseek_spora_delayed_resolution_pilot import OUTPUT as PRIOR
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, save
from spora.historical_resolutions import HistoricalResolutions
from spora.internal_memory import observe_sourced, read_provenance
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-resolution-interference-audit-20261112.json'
LEVELS = (0, 8, 32, 64)


def inspect(model, state, key, expected_current, late_event, late_value):
    selected = select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
    actual_event = selected.get('event_time') if selected['status'] == 'experimental_candidate' else None
    provenance = read_provenance(model, state, key, late_event)
    return {'current_correct': current(model, state, key) == expected_current,
            'late_history_correct': historical(model, state, key, late_event) == late_value,
            'late_provenance': provenance, 'selected': selected,
            'selection_correct': actual_event == late_event,
            'wrong_candidate': actual_event is not None and actual_event != late_event,
            'abstained': selected['status'] == 'abstain',
            'resolution_record': model.historical_resolutions.lookup(key, late_event)}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha or prior['summary']['confirm']['selection_correct'] != 6:
        raise ValueError('Prior development evidence mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    rows = []
    for person in (0, 3):
        key = ENROLLMENTS[person][0]
        distractor = ENROLLMENTS[3 if person == 0 else 0][0]
        noise_keys = [k for k in range(10, 51) if k not in (key, distractor)]
        for scenario_index, scenario in enumerate(SCENARIOS):
            model.historical_resolutions = HistoricalResolutions(model.config.key_count, capacity=4)
            state = fresh_controlled_state(model)
            targets = []
            for event, (subject, value, valid_at) in enumerate(scenario):
                state, decision = observe_sourced(model, state, key if subject == 'T' else distractor,
                                                   value, 'direct', 1.0, valid_at=valid_at)
                if subject == 'T':
                    targets.append((event, value, valid_at))
            late_event, late_value, late_valid = targets[-1]
            expected_current = max(targets, key=lambda x: (x[2], x[0]))[1]
            if sorted(targets, key=lambda x: (x[2], x[0]))[-2][0] != late_event:
                raise ValueError('Experiment requires the delayed event to be the previous effective fact')
            model.historical_resolutions.record(model, state, key, late_event, value=late_value,
                                                valid_at=late_valid, decision='confirm')
            noise_written = 0
            accepted_noise = 0
            for level in LEVELS:
                while noise_written < level:
                    if state.cursor == capacity(model, state):
                        state = grow(model, state)
                    noise_key = noise_keys[noise_written % len(noise_keys)]
                    value = (noise_written * 7 + 2) % model.config.unknown
                    state, decision = observe_sourced(model, state, noise_key, value, 'direct', 1.0)
                    accepted_noise += int(decision['current_write'])
                    noise_written += 1
                metrics = inspect(model, state, key, expected_current, late_event, late_value)
                reload_same = None
                if level == LEVELS[-1]:
                    original_state_digest = frozen_digest(model, state)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / 'individual.pt'
                        save(path, model, state, {'purpose': 'resolution-interference-audit'})
                        restored_model, restored_state, _ = load(path, 'cpu')
                        restored = inspect(restored_model, restored_state, key, expected_current, late_event, late_value)
                        reload_same = (frozen_digest(restored_model, restored_state) == original_state_digest
                                       and restored == metrics)
                rows.append({'person': person, 'scenario': scenario_index, 'noise_writes': level,
                             'accepted_noise_writes': accepted_noise, 'cursor': state.cursor,
                             'page_count': len(state.pages), 'metrics': metrics, 'same_process_reload_same': reload_same})
                print(json.dumps({'person': person, 'scenario': scenario_index, 'noise': level,
                                  'pages': len(state.pages), 'current_correct': metrics['current_correct'],
                                  'history_correct': metrics['late_history_correct'],
                                  'selection_correct': metrics['selection_correct'],
                                  'wrong_candidate': metrics['wrong_candidate'],
                                  'abstained': metrics['abstained'], 'reload_same': reload_same}), flush=True)
    del model.historical_resolutions
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    summary = {}
    for level in LEVELS:
        subset = [r for r in rows if r['noise_writes'] == level]
        summary[str(level)] = {'cases': len(subset),
                               **{field: sum(r['metrics'][field] for r in subset)
                                  for field in ('current_correct', 'late_history_correct', 'selection_correct',
                                                'wrong_candidate', 'abstained')},
                               'same_process_reload_same': sum(r['same_process_reload_same'] is True for r in subset)}
    report = {'status': 'controlled_resolution_interference_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Six reused synthetic delayed-fact trajectories; not a population benchmark.',
                              'Explicit oracle confirmation, numerical writes and preselected irrelevant keys.',
                              'Interference is a deterministic synthetic sequence; more varied histories may differ.',
                              'Selection can overlook neural unknown events and is not a safety certificate.',
                              'Same-process reload only at the final level, not a crash or full-decoder restart.',
                              'No real user individual, natural-language writing or generation evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
