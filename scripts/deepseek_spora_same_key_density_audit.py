"""Audit a dense history where every observation belongs to the same key."""
import json
import statistics
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
from scripts.deepseek_spora_delayed_fact_audit import current, historical, load_individual_from_verified_checkpoint
from scripts.deepseek_spora_indexed_history_audit import OUTPUT as PRIOR, same_neural_state
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, save
from spora.internal_memory import enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-same-key-density-audit-20261112.json'
LEVELS = (2, 8, 16, 32, 64, 128, 256)


def measured(model, state, key):
    outputs, timings = [], []
    for _ in range(3):
        start = time.perf_counter()
        outputs.append(select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1))
        timings.append(time.perf_counter() - start)
    if any(item != outputs[0] for item in outputs):
        raise ValueError('Repeated selection changed')
    return {'selection': outputs[0], 'median_seconds': statistics.median(timings), 'samples_seconds': timings}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    regular = fresh_controlled_state(model)
    indexed = enable_event_key_index(model, fresh_controlled_state(model))
    key = ENROLLMENTS[0][0]
    accepted, deferred, count, rows = 0, 0, 0, []
    for level in LEVELS:
        while count < level:
            if regular.cursor == capacity(model, regular):
                regular, indexed = grow(model, regular), grow(model, indexed)
            value = 1 if count % 2 == 0 else 6
            regular, a = observe_sourced(model, regular, key, value, 'direct', 1.0)
            indexed, b = observe_sourced(model, indexed, key, value, 'direct', 1.0)
            if a != b:
                raise ValueError('Write decisions diverged')
            accepted += int(a['current_write'])
            deferred += int(a['action'] == 'defer')
            count += 1
        if not same_neural_state(regular, indexed):
            raise ValueError('Index changed neural state')
        plain, index = measured(model, regular, key), measured(model, indexed, key)
        expected_event = count - 2
        expected_value = 1 if expected_event % 2 == 0 else 6
        expected_current = 1 if (count - 1) % 2 == 0 else 6
        for result in (plain, index):
            chosen = result['selection']
            result['previous_event_correct'] = (chosen['status'] == 'experimental_candidate'
                                                and chosen.get('event_time') == expected_event)
            result['wrong_candidate'] = (chosen['status'] == 'experimental_candidate'
                                         and chosen.get('event_time') != expected_event)
        rows.append({'writes': count, 'pages': len(indexed.pages), 'accepted_writes': accepted,
                     'deferred_writes': deferred, 'expected_previous_event': expected_event,
                     'oracle_previous_value_correct': historical(model, indexed, key, expected_event) == expected_value,
                     'current_value_correct': current(model, indexed, key) == expected_current,
                     'regular': plain, 'indexed': index})
        print(json.dumps({'writes': count, 'pages': len(indexed.pages), 'accepted': accepted, 'deferred': deferred,
                          'current_correct': rows[-1]['current_value_correct'],
                          'oracle_previous_correct': rows[-1]['oracle_previous_value_correct'],
                          'regular_status': plain['selection']['status'], 'regular_reason': plain['selection'].get('reason'),
                          'regular_correct': plain['previous_event_correct'],
                          'indexed_status': index['selection']['status'], 'indexed_reason': index['selection'].get('reason'),
                          'indexed_correct': index['previous_event_correct'],
                          'indexed_seconds': index['median_seconds']}), flush=True)
    digest = frozen_digest(model, indexed)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'individual.pt'
        save(path, model, indexed, {'purpose': 'same-key-density-audit'})
        restored_model, restored_state, _ = load(path, 'cpu')
        restored_selection = measured(restored_model, restored_state, key)
        reload_same = (frozen_digest(restored_model, restored_state) == digest
                       and restored_selection['selection'] == rows[-1]['indexed']['selection'])
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'controlled_same_key_density_audit', 'rows': rows,
              'same_process_reload_same': reload_same, 'source_checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['One known numeric key, alternating two values and deterministic direct observations.',
                              'Oracle treats every direct observation as a valid successive fact even if controller defers.',
                              'All observations have the same key, so the key index cannot prune events.',
                              'CPU timing includes selector only; three unisolated measurements per point.',
                              'No natural language, independent identities, crash recovery or real user state.',
                              'The exact index records keys as auxiliary metadata, not learned memory weights.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'levels': len(rows), 'same_process_reload_same': reload_same}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
