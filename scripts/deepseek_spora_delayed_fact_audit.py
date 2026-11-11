"""Independent-oracle audit of delayed facts and same-process persistence."""
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
from scripts.spora_event_scan_selector import select_event
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import load, query, save
from spora.integrated_checkpoint import _digest, _validate_binding
from spora.internal_memory import observe_sourced, read_provenance
from spora.language_training import frozen_digest
from spora.named_address import feature_files_sha256
from spora.state_codec import restore

OUTPUT = ROOT / 'runs/deepseek-spora-delayed-fact-audit-v2-20261112.json'
# subject, finite value, effective time. Each last event arrives late; it is
# treated by this test as a truthful direct observation of an earlier time.
SCENARIOS = (
    (('D', 2, 0), ('T', 6, 1), ('T', 1, 0)),
    (('T', 1, 0), ('D', 2, 1), ('T', 6, 2), ('D', 5, 3), ('T', 3, 1)),
    (('T', 6, 0), ('D', 2, 1), ('T', 3, 2), ('D', 5, 3), ('T', 1, 1)),
)


def load_individual_from_verified_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or set(payload) != {'metadata', 'tensors', 'integrity_sha256'}
            or _digest(payload['metadata'], payload['tensors']) != payload['integrity_sha256']):
        raise ValueError('Integrated checkpoint integrity mismatch')
    metadata = payload['metadata']
    config_sha = feature_files_sha256(runtime.MODEL)
    if (metadata.get('base_sha256') != runtime.WEIGHT_SHA
            or runtime.sha(runtime.MODEL / 'model.safetensors') != runtime.WEIGHT_SHA
            or metadata.get('feature_config_sha256') != config_sha):
        raise ValueError('Base provenance mismatch')
    tensors = {name[len('individual/'):]: value for name, value in payload['tensors'].items()
               if name.startswith('individual/')}
    model, state = restore(metadata['individual_schema'], tensors, 'cpu')
    _validate_binding(model, runtime.WEIGHT_SHA, config_sha, metadata.get('feature_variant_sha256'))
    return model, state


def current(model, state, key):
    keys = torch.tensor([key], dtype=torch.long)
    zeros = torch.zeros_like(keys)
    with torch.no_grad():
        return int(query(model, state, keys, zeros, zeros).argmax(-1)[0])


def historical(model, state, key, event):
    keys = torch.tensor([key], dtype=torch.long)
    with torch.no_grad():
        return int(query(model, state, keys, torch.ones_like(keys), torch.full_like(keys, event)).argmax(-1)[0])


def evaluate(model, state, key, expected_latest, expected_previous, late_event, late_value):
    previous = select_event(model, state, key, 'effective_previous', as_of=state.cursor - 1)
    latest = select_event(model, state, key, 'effective_latest', as_of=state.cursor - 1)
    provenance = read_provenance(model, state, key, late_event)
    return {'current_top': current(model, state, key), 'late_history_top': historical(model, state, key, late_event),
            'late_provenance': provenance, 'latest': latest, 'previous': previous,
            'current_correct': current(model, state, key) == expected_latest[1],
            'late_history_correct': historical(model, state, key, late_event) == late_value,
            'latest_event_correct': (latest['status'] == 'experimental_candidate'
                                     and latest.get('event_time') == expected_latest[0]),
            'previous_event_correct': (previous['status'] == 'experimental_candidate'
                                       and previous.get('event_time') == expected_previous[0])}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    before = frozen_digest(model, original)
    rows = []
    for person in (0, 3):
        key = ENROLLMENTS[person][0]
        distractor = ENROLLMENTS[3 if person == 0 else 0][0]
        for scenario_index, scenario in enumerate(SCENARIOS):
            state = fresh_controlled_state(model)
            writes, target_events = [], []
            for event, (subject, value, valid_at) in enumerate(scenario):
                write_key = key if subject == 'T' else distractor
                state, decision = observe_sourced(model, state, write_key, value, 'direct', 1.0, valid_at=valid_at)
                writes.append({'event': event, 'subject': subject, 'value': value, 'valid_at': valid_at,
                               'action': decision['action'], 'current_write': decision['current_write']})
                if subject == 'T':
                    target_events.append((event, value, valid_at))
            # The task's independent oracle treats all direct observations as
            # true, including those that the controller chooses to defer.
            by_effective = sorted(target_events, key=lambda item: (item[2], item[0]))
            expected_latest, expected_previous = by_effective[-1], by_effective[-2]
            late_event, late_value, _ = target_events[-1]
            observation = evaluate(model, state, key, expected_latest, expected_previous, late_event, late_value)
            state_digest = frozen_digest(model, state)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'individual.pt'
                save(path, model, state, {'purpose': 'delayed-fact-temporary-audit'})
                restored_model, restored_state, _ = load(path, 'cpu')
                restored = evaluate(restored_model, restored_state, key, expected_latest, expected_previous,
                                    late_event, late_value)
                reload_same = (frozen_digest(restored_model, restored_state) == state_digest and restored == observation)
            row = {'person': person, 'scenario': scenario_index, 'writes': writes,
                   'expected_latest_event': expected_latest[0], 'expected_latest_value': expected_latest[1],
                   'expected_previous_event': expected_previous[0], 'expected_previous_value': expected_previous[1],
                   'late_event': late_event, 'observed': observation, 'reload_same': reload_same}
            rows.append(row)
            print(json.dumps({'person': person, 'scenario': scenario_index,
                              'late_action': writes[-1]['action'], 'late_current_write': writes[-1]['current_write'],
                              'current_correct': observation['current_correct'],
                              'late_history_correct': observation['late_history_correct'],
                              'previous_event_correct': observation['previous_event_correct'],
                              'previous_status': observation['previous']['status'], 'reload_same': reload_same}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or frozen_digest(model, original) != before:
        raise ValueError('Protected source changed')
    summary = {'cases': len(rows), 'late_deferred': sum(r['writes'][-1]['action'] == 'defer' for r in rows),
               'current_correct': sum(r['observed']['current_correct'] for r in rows),
               'late_history_correct': sum(r['observed']['late_history_correct'] for r in rows),
               'latest_event_correct': sum(r['observed']['latest_event_correct'] for r in rows),
               'previous_event_correct': sum(r['observed']['previous_event_correct'] for r in rows),
               'previous_abstained': sum(r['observed']['previous']['status'] == 'abstain' for r in rows),
               'same_process_reload_same': sum(r['reload_same'] for r in rows)}
    result = {'status': 'independent_oracle_delayed_fact_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Six short synthetic trajectories with two numerical keys; no language model loaded.',
                              'The oracle assumes delayed direct observations are truthful; uncertainty may warrant deferral.',
                              'Direct history reads use known event positions; selector must infer them independently.',
                              'Checkpoint restored from verified integrated tensors; default user state is untouched.',
                              'Reload is same-process, not a process restart or crash-recovery test.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
