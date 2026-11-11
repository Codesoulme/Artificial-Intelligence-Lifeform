"""Validate independent provenance ridge on new keys and two-page histories."""
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
from scripts.deepseek_spora_provenance_ridge_pilot import OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, query
from spora.internal_memory import ACTIONS, SOURCES, _decode_provenance, enable_event_key_index, observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/deepseek-spora-provenance-ridge-holdout-20261112.json'
RIDGES = (0.01, 0.003)
TRAJECTORIES = tuple(('dense', k) for k in (2, 5, 7, 8, 9, 10, 11, 12)) + tuple(
    ('mixed', s) for s in (2017, 3301, 4421, 5897, 6949, 7907, 8951, 9973))
NOISE = (0.0, 0.05, 0.1, 0.2)


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
        rng = random.Random(873113 + seed)
        state = enable_event_key_index(model, fresh_controlled_state(model))
        matrices, precision = {r: [] for r in RIDGES}, {r: [] for r in RIDGES}
        addresses, writes = [], []
        for event in range(128):
            if state.cursor == capacity(model, state):
                state = grow(model, state)
            page, local = divmod(event, model.config.time_count)
            if local == 0:
                for ridge in RIDGES:
                    matrices[ridge].append(torch.zeros(27, width))
                    precision[ridge].append(torch.eye(width) / ridge)
            key = seed if kind == 'dense' else rng.randrange(model.config.key_count)
            value = rng.randrange(model.config.unknown)
            source = rng.choice(SOURCES)
            quality = rng.choice((0.0, 0.33, 0.67, 1.0))
            valid_at = event - rng.randrange(min(event, 16) + 1)
            keys = torch.tensor([key], dtype=torch.long)
            modes = torch.ones_like(keys)
            times = torch.tensor([local], dtype=torch.long)
            _, bound = model.representation(keys, modes, times)
            address = model.memory_address(model.memories[-1], bound, keys, modes)[0].detach()
            state, decision = observe_sourced(model, state, key, value, source, quality, valid_at=valid_at)
            quality_code = round(quality * 255)
            bits = [(quality_code >> i) & 1 for i in range(8)] + [(valid_at >> i) & 1 for i in range(10)]
            target = torch.cat((torch.nn.functional.one_hot(torch.tensor(SOURCES.index(source)), 4),
                                torch.nn.functional.one_hot(torch.tensor(ACTIONS.index(decision['action'])), 5),
                                torch.tensor(bits))).float()
            for ridge in RIDGES:
                p = precision[ridge][page]
                projected = p @ address
                gain = projected / (1 + address @ projected)
                matrices[ridge][page] += (target - matrices[ridge][page] @ address)[:, None] * gain[None]
                p = p - projected[:, None] * gain[None, :]
                precision[ridge][page] = (p + p.T) * 0.5
                if not bool(torch.isfinite(matrices[ridge][page]).all() and torch.isfinite(precision[ridge][page]).all()):
                    raise ValueError('Nonfinite RLS candidate')
            addresses.append(address)
            writes.append({'event': event, 'key': key, 'value': value, 'source': source,
                           'quality_code': quality_code, 'valid_at': valid_at, 'action': decision['action']})
        baseline_error = max(float((matrices[0.01][p] - state.internal.provenance[p]).abs().max()) for p in range(2))
        if baseline_error > 0.0001:
            raise ValueError(f'Independent baseline differs from actual writer: {baseline_error}')
        addresses = torch.stack(addresses)
        keys = torch.tensor([item['key'] for item in writes], dtype=torch.long)
        events = torch.arange(128, dtype=torch.long)
        predicted = query(model, state, keys, torch.ones_like(keys), events).argmax(-1)
        grouped = []
        for ridge in RIDGES:
            generator = torch.Generator(device='cpu').manual_seed(691997 + seed)
            noise = [torch.randn(matrix.shape, generator=generator, dtype=matrix.dtype) for matrix in matrices[ridge]]
            rms = [float(matrix.square().mean().sqrt()) for matrix in matrices[ridge]]
            conditions = []
            for level in NOISE:
                raw = torch.cat([addresses[p*64:(p+1)*64] @ (matrices[ridge][p] + level * rms[p] * noise[p]).T
                                 for p in range(2)])
                accepted = correct = wrong = 0
                for event in range(128):
                    output = _decode_provenance(raw[event], event, 2)
                    known = int(predicted[event]) != model.config.unknown
                    selected = known and 'reason' not in output
                    expected = writes[event]
                    exact = (output.get('source') == expected['source'] and output.get('action') == expected['action']
                             and output.get('valid_at') == expected['valid_at']
                             and output.get('quality') == expected['quality_code'] / 255
                             and int(predicted[event]) == expected['value'])
                    accepted += int(selected)
                    correct += int(selected and exact)
                    wrong += int(selected and not exact)
                conditions.append({'relative_rms_noise': level, 'accepted': accepted,
                                   'accepted_correct': correct, 'accepted_incorrect': wrong})
            grouped.append({'ridge': ridge,
                            'precision_min_eigenvalue': min(float(torch.linalg.eigvalsh(p).min()) for p in precision[ridge]),
                            'conditions': conditions})
        row = {'kind': kind, 'seed_or_key': seed, 'baseline_matrix_max_error': baseline_error, 'ridges': grouped}
        trajectories.append(row)
        print(json.dumps({'kind': kind, 'seed_or_key': seed,
                          'clean': {str(g['ridge']): g['conditions'][0] for g in grouped}}), flush=True)
    if runtime.sha(runtime.CHECKPOINT) != source_sha or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected source changed')
    report = {'status': 'two_page_ridge_development_holdout', 'trajectories': trajectories,
              'source_checkpoint_sha256': source_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Sixteen synthetic two-page trajectories, with keys and seeds distinct from pilot.',
                              'Candidate ridge chosen after pilot; further development evidence, not a final benchmark.',
                              'Random numeric values and sources; no language or autonomous learning.',
                              'Expected actions are model controller outputs; action quality not evaluated.',
                              'Gaussian tensor noise is artificial and does not simulate actual sleep or corruption.',
                              'Separate provenance precision adds memory and persistence requirements; not yet integrated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'trajectories': len(trajectories)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
