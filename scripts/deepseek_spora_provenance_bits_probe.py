"""Measure raw provenance-bit decoding without changing selection thresholds."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_delayed_fact_audit import load_individual_from_verified_checkpoint
from scripts.deepseek_spora_same_key_retention_probe import OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import ACTIONS, SOURCES, enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-provenance-bits-probe-20261112.json'
BANDS = ((0.2, 0.8), (0.3, 0.7), (0.4, 0.6))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    previous = json.loads(PRIOR.read_text(encoding='utf-8'))
    if previous['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    trajectories = []
    for person in (0, 3):
        key = ENROLLMENTS[person][0]
        for pattern in ('alternating', 'three_values', 'mixed_sources'):
            state = enable_event_key_index(model, fresh_controlled_state(model))
            writes = []
            for event in range(64):
                value = ((1, 6)[event % 2] if pattern == 'alternating' else (1, 3, 6)[event % 3])
                source = (('direct', 'report', 'inference', 'simulation')[event % 4]
                          if pattern == 'mixed_sources' else 'direct')
                state, decision = observe_sourced(model, state, key, value, source, 1.0)
                writes.append({'event': event, 'value': value, 'source': source, 'action': decision['action']})
            rows = []
            for start in range(0, 64, 16):
                events = list(range(start, start + 16))
                keys = torch.full((len(events),), key, dtype=torch.long)
                times = torch.tensor(events, dtype=torch.long)
                modes = torch.ones_like(keys)
                with torch.no_grad():
                    known = query(model, state, keys, modes, times).argmax(-1) != model.config.unknown
                    _, bound = model.representation(keys, modes, times % model.config.time_count)
                    addresses = model.memory_address(model.memories[-1], bound, keys, modes)
                    raw = addresses @ state.internal.provenance[0].T
                for j, event in enumerate(events):
                    source_scores, action_scores = raw[j, :4], raw[j, 4:9]
                    bits = raw[j, 9:] / source_scores.sum().clamp_min(0.1)
                    bit_values = (bits > 0.5).tolist()
                    quality_code = sum(int(bit) << i for i, bit in enumerate(bit_values[:8]))
                    valid_at = sum(int(bit) << i for i, bit in enumerate(bit_values[8:]))
                    decoded_source = SOURCES[int(source_scores.argmax())]
                    decoded_action = ACTIONS[int(action_scores.argmax())]
                    base_confident = bool(known[j]) and float(source_scores.max()) >= 0.7 and float(action_scores.max()) >= 0.7
                    exact = (decoded_source == writes[event]['source'] and decoded_action == writes[event]['action']
                             and quality_code == 255 and valid_at == event)
                    row = {'event': event, 'known': bool(known[j]), 'expected': writes[event],
                           'decoded_source': decoded_source, 'decoded_action': decoded_action,
                           'decoded_quality_code': quality_code, 'decoded_valid_at': valid_at,
                           'base_confident': base_confident, 'exact': exact,
                           'bits': [float(value) for value in bits]}
                    for low, high in BANDS:
                        confident = base_confident and not bool(((bits > low) & (bits < high)).any())
                        row[f'confident_{low}_{high}'] = confident
                    rows.append(row)
            summary = {'person': person, 'pattern': pattern, 'events': len(rows),
                       'naive_exact': sum(r['exact'] for r in rows),
                       **{f'band_{low}_{high}': {'accepted': sum(r[f'confident_{low}_{high}'] for r in rows),
                                                 'accepted_incorrect': sum(r[f'confident_{low}_{high}'] and not r['exact'] for r in rows)}
                          for low, high in BANDS}}
            trajectories.append({'summary': summary, 'rows': rows})
            print(json.dumps(summary), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'raw_provenance_bit_calibration_probe', 'trajectories': trajectories,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Six synthetic trajectories, 64 events each, using two known numeric keys.',
                              'Effective time always equals observation time; no delayed facts or corrupted inputs.',
                              'Expected action comes from the model controller; this does not validate controller decisions.',
                              'Expected source and time come from test inputs, not inferred real-world truth.',
                              'No train/holdout tuning protocol; thresholds are diagnostic only, not proposed for deployment.',
                              'No language model, user data, persistence or end-to-end answers evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'trajectories': len(trajectories)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
