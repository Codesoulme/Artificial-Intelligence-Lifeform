"""Stress candidate provenance bands with synthetic tensor perturbations."""
import argparse
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
from scripts.deepseek_spora_provenance_band_selector_pilot import OUTPUT as PRIOR
from scripts.deepseek_spora_provenance_bits_probe import BANDS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import ACTIONS, SOURCES, enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-provenance-noise-stress-20261112.json'
HIGH_OUTPUT = ROOT / 'runs/deepseek-spora-provenance-noise-stress-high-20261112.json'
SEEDS = (4111, 8209, 12109, 17317)
LEVELS = (0.0, 0.01, 0.05, 0.1, 0.2)
HIGH_LEVELS = (0.3, 0.4, 0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high', action='store_true')
    args = parser.parse_args()
    output_path = HIGH_OUTPUT if args.high else OUTPUT
    levels = HIGH_LEVELS if args.high else LEVELS
    if output_path.exists():
        raise FileExistsError(output_path)
    source_sha = runtime.sha(runtime.CHECKPOINT)
    if json.loads(PRIOR.read_text(encoding='utf-8'))['source_checkpoint_sha256'] != source_sha:
        raise ValueError('Prior checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original = load_individual_from_verified_checkpoint(runtime.CHECKPOINT)
    original_digest = frozen_digest(model, original)
    results = []
    for seed in SEEDS:
        rng = random.Random(seed)
        state = enable_event_key_index(model, fresh_controlled_state(model))
        writes = []
        for event in range(64):
            key = rng.randrange(model.config.key_count)
            value = rng.randrange(model.config.unknown)
            source = rng.choice(SOURCES)
            quality = rng.choice((0.2, 0.5, 0.75, 1.0))
            valid_at = event - rng.randrange(min(event, 12) + 1)
            state, decision = observe_sourced(model, state, key, value, source, quality, valid_at=valid_at)
            writes.append({'event': event, 'key': key, 'value': value, 'source': source,
                           'quality_code': round(quality * 255), 'valid_at': valid_at, 'action': decision['action']})
        pristine = state.internal.provenance[0]
        rng_torch = torch.Generator(device='cpu').manual_seed(seed)
        perturbation = torch.randn(pristine.shape, generator=rng_torch, dtype=pristine.dtype)
        rms = float(pristine.square().mean().sqrt())
        keys = torch.tensor([w['key'] for w in writes], dtype=torch.long)
        events = torch.arange(64, dtype=torch.long)
        modes = torch.ones_like(keys)
        with torch.no_grad():
            predicted = query(model, state, keys, modes, events).argmax(-1)
            _, bound = model.representation(keys, modes, events % model.config.time_count)
            addresses = model.memory_address(model.memories[-1], bound, keys, modes)
        for level in levels:
            with torch.no_grad():
                raw = addresses @ (pristine + level * rms * perturbation).T
            rows = []
            for event in range(64):
                source, action = raw[event, :4], raw[event, 4:9]
                bits = raw[event, 9:] / source.sum().clamp_min(0.1)
                code = (bits > 0.5).tolist()
                quality_code = sum(int(v) << i for i, v in enumerate(code[:8]))
                valid_at = sum(int(v) << i for i, v in enumerate(code[8:]))
                source_label, action_label = SOURCES[int(source.argmax())], ACTIONS[int(action.argmax())]
                value = int(predicted[event])
                base = value != model.config.unknown and float(source.max()) >= 0.7 and float(action.max()) >= 0.7
                valid = valid_at <= event
                expected = writes[event]
                exact = (source_label == expected['source'] and action_label == expected['action']
                         and quality_code == expected['quality_code'] and valid_at == expected['valid_at']
                         and value == expected['value'])
                row = {'event': event, 'expected': expected, 'predicted_value': value, 'decoded_valid_at': valid_at,
                       'decoded_quality_code': quality_code, 'decoded_source': source_label,
                       'decoded_action': action_label, 'exact': exact}
                for low, high in BANDS:
                    row[f'accepted_{low}_{high}'] = base and valid and not bool(((bits > low) & (bits < high)).any())
                rows.append(row)
            summary = {'seed': seed, 'relative_rms_noise': level, 'events': len(rows),
                       **{f'band_{low}_{high}': {'accepted': sum(r[f'accepted_{low}_{high}'] for r in rows),
                                                 'accepted_incorrect': sum(r[f'accepted_{low}_{high}'] and not r['exact'] for r in rows)}
                          for low, high in BANDS}}
            results.append({'summary': summary, 'rows': rows})
            print(json.dumps(summary), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'synthetic_provenance_noise_stress', 'results': results,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Four deterministic 64-event synthetic interleaved trajectories.',
                              'Noise is Gaussian relative to tensor RMS; not a model of actual sleep compression or corruption.',
                              'Tests provenance tensor perturbation only; memory values and key index stay unchanged.',
                              'Expected action is controller output; source/quality/time are explicit test inputs.',
                              'Thresholds were investigated after development; no production threshold is changed.',
                              'No natural-language operation or real individual evaluated.']}
    with output_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'conditions': len(results)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
