"""Probe raw provenance thresholds on new keys and varied event metadata."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_delayed_fact_audit import load_individual_from_verified_checkpoint
from scripts.deepseek_spora_provenance_bits_probe import BANDS, OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import ACTIONS, SOURCES, enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-provenance-holdout-probe-20261112.json'
KEYS = (2, 9, 18, 27, 35, 45, 52, 61)
SEED = 672019


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    if json.loads(PRIOR.read_text(encoding='utf-8'))['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    trajectories = []
    for key in KEYS:
        if key >= model.config.key_count:
            raise ValueError('Holdout key out of range')
        rng = random.Random(SEED + key)
        state = enable_event_key_index(model, fresh_controlled_state(model))
        writes = []
        for event in range(64):
            value = rng.randrange(model.config.unknown)
            source = rng.choice(SOURCES)
            quality = rng.choice((0.2, 0.5, 0.75, 1.0))
            valid_at = event - rng.randrange(min(event, 12) + 1)
            state, decision = observe_sourced(model, state, key, value, source, quality, valid_at=valid_at)
            writes.append({'event': event, 'value': value, 'source': source, 'quality_code': round(quality * 255),
                           'valid_at': valid_at, 'action': decision['action']})
        rows = []
        for start in range(0, 64, 16):
            events = list(range(start, start + 16))
            keys = torch.full((len(events),), key, dtype=torch.long)
            times = torch.tensor(events, dtype=torch.long)
            modes = torch.ones_like(keys)
            with torch.no_grad():
                predicted = query(model, state, keys, modes, times).argmax(-1)
                _, bound = model.representation(keys, modes, times % model.config.time_count)
                addresses = model.memory_address(model.memories[-1], bound, keys, modes)
                raw = addresses @ state.internal.provenance[0].T
            for j, event in enumerate(events):
                source_scores, action_scores = raw[j, :4], raw[j, 4:9]
                bits = raw[j, 9:] / source_scores.sum().clamp_min(0.1)
                bit_values = (bits > 0.5).tolist()
                quality_code = sum(int(bit) << i for i, bit in enumerate(bit_values[:8]))
                valid_at = sum(int(bit) << i for i, bit in enumerate(bit_values[8:]))
                source = SOURCES[int(source_scores.argmax())]
                action = ACTIONS[int(action_scores.argmax())]
                value = int(predicted[j])
                base_confident = value != model.config.unknown and float(source_scores.max()) >= 0.7 and float(action_scores.max()) >= 0.7
                exact_metadata = (source == writes[event]['source'] and action == writes[event]['action']
                                  and quality_code == writes[event]['quality_code'] and valid_at == writes[event]['valid_at'])
                exact_all = exact_metadata and value == writes[event]['value']
                row = {'event': event, 'expected': writes[event], 'predicted_value': value,
                       'decoded_source': source, 'decoded_action': action, 'decoded_quality_code': quality_code,
                       'decoded_valid_at': valid_at, 'metadata_exact': exact_metadata, 'all_exact': exact_all,
                       'base_confident': base_confident}
                for low, high in BANDS:
                    row[f'confident_{low}_{high}'] = base_confident and not bool(((bits > low) & (bits < high)).any())
                rows.append(row)
        summary = {'key': key, 'events': len(rows), 'metadata_exact': sum(r['metadata_exact'] for r in rows),
                   'all_exact': sum(r['all_exact'] for r in rows),
                   **{f'band_{low}_{high}': {'accepted': sum(r[f'confident_{low}_{high}'] for r in rows),
                                             'metadata_incorrect': sum(r[f'confident_{low}_{high}'] and not r['metadata_exact'] for r in rows),
                                             'any_incorrect': sum(r[f'confident_{low}_{high}'] and not r['all_exact'] for r in rows)}
                      for low, high in BANDS}}
        trajectories.append({'summary': summary, 'rows': rows})
        print(json.dumps(summary), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'varied_metadata_development_holdout', 'seed': SEED, 'trajectories': trajectories,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Eight new numerical keys in one deterministic test, 64 writes each.',
                              'Synthetic random source, quality and past effective time; no identity or language evaluation.',
                              'Expected action uses the actual controller decision; action quality not assessed.',
                              'Confidence bands were explored on prior examples, so this is not a final independent benchmark.',
                              'Base model pretraining and related individual training may cover these finite values and keys.',
                              'No thresholds or production behavior are changed by this probe.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'trajectories': len(trajectories)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
