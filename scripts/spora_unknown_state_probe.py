"""Evaluate native known/unknown decisions across a fresh controlled trajectory."""
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.growing import load, new_state, query, save
from spora.internal_memory import InternalState, PROVENANCE_WIDTH, observe_sourced
from spora.language_training import frozen_digest
from spora.online_time import OnlineState

OUTPUT = ROOT / 'runs/spora-unknown-fresh-state-reload-20261109.json'
CHECKPOINT = ROOT / 'runs/spora-unknown-fresh-state-reload-20261109.pt'


def fresh_controlled_state(model):
    state = new_state(model)
    width = model.config.memory_width
    device = next(model.parameters()).device
    internal = InternalState(torch.zeros(3, width, device=device),
                             [torch.zeros(PROVENANCE_WIDTH, width, device=device) for _ in state.pages],
                             state.cursor)
    result = replace(state, internal=internal, runtime=OnlineState() if hasattr(model, 'time_world') else None)
    result.internal.validate(model, result)
    return result


@torch.no_grad()
def measure(model, state, expected):
    keys = torch.arange(model.config.key_count, dtype=torch.long, device=next(model.parameters()).device)
    zeros = torch.zeros_like(keys)
    predicted = query(model, state, keys, zeros, zeros).argmax(-1).tolist()
    unknown = model.config.unknown
    rows = []
    for key, value in enumerate(predicted):
        truth = expected.get(key, unknown)
        rows.append({'key': key, 'expected': truth, 'predicted': value,
                     'known_expected': truth != unknown, 'known_predicted': value != unknown})
    return {'examples': len(rows), 'known': len(expected),
            'false_known': sum(not r['known_expected'] and r['known_predicted'] for r in rows),
            'false_unknown': sum(r['known_expected'] and not r['known_predicted'] for r in rows),
            'known_value_correct': sum(r['known_expected'] and r['expected'] == r['predicted'] for r in rows),
            'rows': rows}


def run():
    if OUTPUT.exists() or CHECKPOINT.exists():
        raise FileExistsError('Fresh-state output already exists')
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Original individual mismatch')
    torch.set_num_threads(2)
    rng = random.Random(4612)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    state = fresh_controlled_state(model)
    expected, actions, phases = {}, [], {}
    phases['empty'] = measure(model, state, expected)
    order = list(range(model.config.key_count))
    rng.shuffle(order)
    for phase, keys in (('new', order[:16]), ('corrected', order[:8]), ('interference', order[16:32])):
        for key in keys:
            previous = expected.get(key)
            value = rng.randrange(model.config.unknown)
            if previous is not None and value == previous:
                value = (value + 1) % model.config.unknown
            state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
            expected[key] = value
            actions.append({'phase': phase, 'key': key, 'value': value, 'action': decision['action'],
                            'current_write': decision['current_write']})
        phases[phase] = measure(model, state, expected)
    expected_digest = frozen_digest(model, state)
    save(CHECKPOINT, model, state, {'purpose': 'fresh-state-unknown-probe'})
    reloaded_model, reloaded_state, _ = load(CHECKPOINT, 'cpu')
    reloaded_digest = frozen_digest(reloaded_model, reloaded_state)
    if reloaded_digest != expected_digest:
        raise ValueError('Reloaded state differs from saved state')
    phases['reloaded'] = measure(reloaded_model, reloaded_state, expected)
    if phases['reloaded']['rows'] != phases['interference']['rows']:
        raise ValueError('Reloaded native predictions differ')
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Original individual changed')
    report = {'status': 'fresh_controlled_state_development_only', 'individual_sha256': INDIVIDUAL_SHA,
              'seed': 4612, 'actions': actions, 'phases': phases, 'checkpoint_sha256': sha(CHECKPOINT),
              'checkpoint_frozen_digest': expected_digest,
              'limitations': ['Fresh state uses the same trained model and address basis, not an independently trained individual.',
                              'Direct observations are treated as ground truth even when the write controller defers.',
                              'Only finite 64-key/16-value current memory is tested.',
                              'Does not test natural-language routing, historical queries, or crash recovery.',
                              'Reload occurs within one process; it is not a crash recovery test.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({phase: {k: v for k, v in value.items() if k != 'rows'}
                      for phase, value in phases.items()}))


if __name__ == '__main__':
    run()
