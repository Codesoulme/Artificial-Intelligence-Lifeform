"""Audit native finite-memory presence after writes, corrections and reload.

Truth derives from explicitly committed current writes, not inferred facts.
The numeric key is supplied by the experiment; no language router is tested.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, query, save
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-presence-multiseed-reload-20261114.json'
SEEDS = (1907, 1913, 1931)


@torch.no_grad()
def measure(model, state, expected):
    device = next(model.parameters()).device
    keys = torch.arange(model.config.key_count, dtype=torch.long, device=device)
    zeros = torch.zeros_like(keys)
    logits = query(model, state, keys, zeros, zeros).float()
    if logits.shape != (model.config.key_count, model.config.value_count) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Invalid current memory logits')
    probabilities = logits.softmax(-1)
    predictions = probabilities.argmax(-1)
    rows = []
    for key in range(model.config.key_count):
        value = int(predictions[key])
        expected_value = expected.get(key)
        top = probabilities[key].topk(2).values
        rows.append({'key': key, 'expected_value': expected_value, 'predicted_value': value,
                     'expected_known': expected_value is not None,
                     'predicted_known': value != model.config.unknown,
                     'unknown_probability': float(probabilities[key, model.config.unknown]),
                     'top_probability': float(top[0]), 'margin': float(top[0] - top[1])})
    return {'rows': rows,
            'summary': {'keys': len(rows), 'known': sum(r['expected_known'] for r in rows),
                        'false_known': sum(not r['expected_known'] and r['predicted_known'] for r in rows),
                        'false_unknown': sum(r['expected_known'] and not r['predicted_known'] for r in rows),
                        'known_value_correct': sum(r['expected_known'] and r['predicted_value'] == r['expected_value'] for r in rows),
                        'wrong_known_value': sum(r['expected_known'] and r['predicted_known']
                                                 and r['predicted_value'] != r['expected_value'] for r in rows)}}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected individual digest mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    model, original, _ = load(INDIVIDUAL, 'cpu')
    original_digest = frozen_digest(model, original)
    trials = []
    with tempfile.TemporaryDirectory(prefix='spora-presence-') as folder:
        for seed in SEEDS:
            rng = random.Random(seed)
            state = fresh_controlled_state(model)
            expected = {}
            actions = []
            phases = {'empty': measure(model, state, expected)}
            keys = list(range(model.config.key_count))
            rng.shuffle(keys)
            for phase, sequence in (('initial', keys[:16]), ('correction', keys[:8]),
                                    ('interference', [keys[16 + (i % 16)] for i in range(48)])):
                for key in sequence:
                    if state.cursor == capacity(model, state):
                        state = grow(model, state)
                    old = expected.get(key)
                    value = rng.randrange(model.config.unknown)
                    if old is not None and value == old:
                        value = (value + 1) % model.config.unknown
                    state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
                    if decision['current_write']:
                        expected[key] = value
                    actions.append({'phase': phase, 'key': key, 'proposed_value': value,
                                    'action': decision['action'], 'current_write': decision['current_write']})
                phases[phase] = measure(model, state, expected)
            checkpoint = Path(folder) / f'state-{seed}.pt'
            save(checkpoint, model, state, {'purpose': 'presence-multiseed-isolated-audit'})
            checkpoint_sha = sha(checkpoint)
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--reload', str(checkpoint)],
                                     capture_output=True, text=True, encoding='utf-8', check=True, timeout=90)
            restored = json.loads(process.stdout)
            prior = phases['interference']
            matched = (restored['predicted_values'] == [row['predicted_value'] for row in prior['rows']]
                       and restored['digest'] == frozen_digest(model, state))
            if not matched or sha(checkpoint) != checkpoint_sha:
                raise ValueError('Independent process did not restore equivalent state')
            trials.append({'seed': seed, 'actions': actions, 'phases': phases,
                           'cursor': state.cursor, 'pages': len(state.pages),
                           'independent_reload_matches': matched, 'checkpoint_sha256': checkpoint_sha})
            print(json.dumps({'seed': seed, 'pages': len(state.pages),
                              'summary': {key: value['summary'] for key, value in phases.items()},
                              'independent_reload_matches': matched}), flush=True)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != original_digest:
        raise ValueError('Protected individual changed')
    report = {'status': 'finite_presence_controlled_multiseed_audit', 'trials': trials,
              'source_individual_sha256': INDIVIDUAL_SHA, 'seconds': time.perf_counter()-started,
              'limitations': ['Three random trajectories share one trained finite-symbol individual.',
                              'Expected state follows accepted direct current writes, not independent truth adjudication.',
                              '64 supplied numerical keys and 16 finite values; no natural-language identity selection.',
                              'Presence is argmax, and probabilities are uncalibrated.',
                              'Cross-process restart is not a crash-recovery or concurrent-write test.',
                              'No direct connection to online low-rank candidates or autonomous conversation.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def reload(path):
    torch.set_num_threads(2)
    model, state, _ = load(path, 'cpu')
    result = measure(model, state, {})
    print(json.dumps({'digest': frozen_digest(model, state),
                      'predicted_values': [row['predicted_value'] for row in result['rows']]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reload', type=Path)
    args = parser.parse_args()
    reload(args.reload) if args.reload else run()
