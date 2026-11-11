"""Compare independent provenance RLS ridges on identical controlled writes."""
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
from scripts.deepseek_spora_provenance_noise_stress import HIGH_OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import ACTIONS, SOURCES, _decode_provenance, enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-provenance-ridge-pilot-20261112.json'
RIDGES = (0.01, 0.003, 0.001, 0.0001)
TRAJECTORIES = (('dense', 0), ('dense', 1), ('dense', 3), ('dense', 4),
                ('mixed', 1801), ('mixed', 2707), ('mixed', 4201), ('mixed', 7307))
NOISE = (0.0, 0.1, 0.2)


@torch.no_grad()
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
    if model.config.history_ridge != 0.01 or model.config.time_count != 64:
        raise ValueError('Unexpected historical configuration')
    width = model.config.memory_width
    trajectories = []
    for kind, seed in TRAJECTORIES:
        rng = random.Random(992771 + seed)
        state = enable_event_key_index(model, fresh_controlled_state(model))
        matrices = {ridge: torch.zeros(27, width) for ridge in RIDGES}
        precision = {ridge: torch.eye(width) / ridge for ridge in RIDGES}
        addresses, writes = [], []
        for event in range(64):
            key = seed if kind == 'dense' else rng.randrange(model.config.key_count)
            value = rng.randrange(model.config.unknown)
            source = rng.choice(SOURCES)
            quality = rng.choice((0.2, 0.5, 0.75, 1.0))
            valid_at = event - rng.randrange(min(event, 12) + 1)
            keys = torch.tensor([key], dtype=torch.long)
            modes = torch.ones_like(keys)
            times = torch.tensor([event], dtype=torch.long)
            _, bound = model.representation(keys, modes, times)
            address = model.memory_address(model.memories[-1], bound, keys, modes)[0].detach()
            state, decision = observe_sourced(model, state, key, value, source, quality, valid_at=valid_at)
            quality_code = round(quality * 255)
            bits = [(quality_code >> i) & 1 for i in range(8)] + [(valid_at >> i) & 1 for i in range(10)]
            target = torch.cat((torch.nn.functional.one_hot(torch.tensor(SOURCES.index(source)), 4),
                                torch.nn.functional.one_hot(torch.tensor(ACTIONS.index(decision['action'])), 5),
                                torch.tensor(bits))).float()
            for ridge in RIDGES:
                p = precision[ridge]
                projected = p @ address
                gain = projected / (1 + address @ projected)
                matrices[ridge] = matrices[ridge] + (target - matrices[ridge] @ address)[:, None] * gain[None]
                p = p - projected[:, None] * gain[None, :]
                precision[ridge] = (p + p.T) * 0.5
                if not bool(torch.isfinite(matrices[ridge]).all() and torch.isfinite(precision[ridge]).all()):
                    raise ValueError('Nonfinite RLS candidate')
            addresses.append(address)
            writes.append({'event': event, 'key': key, 'value': value, 'source': source,
                           'quality_code': quality_code, 'valid_at': valid_at, 'action': decision['action']})
        baseline_error = float((matrices[0.01] - state.internal.provenance[0]).abs().max())
        if baseline_error > 0.0001:
            raise ValueError(f'Independent baseline differs from actual writer: {baseline_error}')
        addresses = torch.stack(addresses)
        keys = torch.tensor([item['key'] for item in writes], dtype=torch.long)
        events = torch.arange(64, dtype=torch.long)
        predicted = query(model, state, keys, torch.ones_like(keys), events).argmax(-1)
        grouped = []
        for ridge in RIDGES:
            matrix = matrices[ridge]
            rms = float(matrix.square().mean().sqrt())
            generator = torch.Generator(device='cpu').manual_seed(181001 + seed)
            noise = torch.randn(matrix.shape, generator=generator, dtype=matrix.dtype)
            conditions = []
            for level in NOISE:
                raw = addresses @ (matrix + level * rms * noise).T
                rows = []
                for event in range(64):
                    output = _decode_provenance(raw[event], event, 2)
                    known = int(predicted[event]) != model.config.unknown
                    accepted = known and 'reason' not in output
                    expected = writes[event]
                    exact = (output.get('source') == expected['source'] and output.get('action') == expected['action']
                             and output.get('valid_at') == expected['valid_at']
                             and output.get('quality') == expected['quality_code'] / 255
                             and int(predicted[event]) == expected['value'])
                    rows.append({'event': event, 'accepted': accepted, 'exact': exact})
                conditions.append({'relative_rms_noise': level, 'accepted': sum(r['accepted'] for r in rows),
                                   'accepted_correct': sum(r['accepted'] and r['exact'] for r in rows),
                                   'accepted_incorrect': sum(r['accepted'] and not r['exact'] for r in rows)})
            grouped.append({'ridge': ridge, 'matrix_rms': rms,
                            'precision_min_eigenvalue': float(torch.linalg.eigvalsh(precision[ridge]).min()),
                            'conditions': conditions})
        row = {'kind': kind, 'seed_or_key': seed, 'baseline_matrix_max_error': baseline_error, 'ridges': grouped}
        trajectories.append(row)
        print(json.dumps({'kind': kind, 'seed_or_key': seed, 'baseline_matrix_max_error': baseline_error,
                          'clean': {str(g['ridge']): g['conditions'][0] for g in grouped}}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'provenance_independent_ridge_pilot', 'trajectories': trajectories,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Eight synthetic one-page trajectories and four ridge settings; all development data.',
                              'Separate provenance matrices and inverse matrices exist only in experiment memory.',
                              'Expected source, quality and time come from generated inputs; action from model controller.',
                              'Uncalibrated threshold remains unchanged; Gaussian noise is artificial, not actual sleep.',
                              'Candidate would require additional state and safe persistence, which are not implemented.',
                              'No full decoder, identity resolution, natural-language writing or real user data.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'trajectories': len(trajectories)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
