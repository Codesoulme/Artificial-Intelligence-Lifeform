"""Measure history known/unknown margins on separate finite trajectories."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-history-novelty-checkpoints-20261109.json'
CALIBRATION = (4615, 4616, 4617)
VALIDATION = (4621, 4622, 4623)
CHECKPOINTS = (8, 32, 64, 96, 192)


@torch.no_grad()
def trajectory(model, seed):
    rng = random.Random(seed)
    state = fresh_controlled_state(model)
    order = list(range(model.config.key_count))
    rng.shuffle(order)
    expected, events, results = {}, [], []
    for step in range(192):
        if state.cursor >= capacity(model, state):
            state = grow(model, state)
        key = order[step] if step < len(order) else rng.choice(order)
        previous = expected.get(key)
        value = rng.randrange(model.config.unknown)
        if previous is not None and value == previous:
            value = (value + 1) % model.config.unknown
        state, _ = observe_sourced(model, state, key, value, 'direct', 1.0)
        expected[key] = value
        events.append((key, value))
        if step + 1 in CHECKPOINTS:
            results.append(measure(model, state, seed, events))
    return results


@torch.no_grad()
def measure(model, state, seed, events):
    allocated = capacity(model, state)
    keys = torch.arange(model.config.key_count).repeat(allocated)
    times = torch.arange(allocated).repeat_interleave(model.config.key_count)
    margins, predictions = [], []
    for start in range(0, len(keys), 256):
        k, t = keys[start:start + 256], times[start:start + 256]
        logits = query(model, state, k, torch.ones_like(k), t)
        margins.append((logits[:, -1] - logits[:, :-1].max(dim=-1).values).cpu())
        predictions.append(logits.argmax(dim=-1).cpu())
    margins, predictions = torch.cat(margins), torch.cat(predictions)
    padded = events + [(-1, -1)] * (allocated - len(events))
    true_keys = torch.tensor([event[0] for event in padded]).repeat_interleave(model.config.key_count)
    true_values = torch.tensor([event[1] for event in padded]).repeat_interleave(model.config.key_count)
    present = keys == true_keys
    return {'seed': seed, 'cursor': state.cursor, 'margins': margins, 'predictions': predictions, 'present': present,
            'true_values': true_values, 'keys': keys, 'times': times}


def summarize(data, threshold):
    present = data['present']
    known = data['margins'] < threshold
    false_known = (~present) & known
    false_unknown = present & ~known
    value_correct = present & known & (data['predictions'] == data['true_values'])
    # For a nonzero threshold, use the best value class rather than the baseline argmax.
    return {'seed': data['seed'], 'cursor': data['cursor'], 'present': int(present.sum()), 'absent': int((~present).sum()),
            'false_known': int(false_known.sum()), 'false_unknown': int(false_unknown.sum()),
            'baseline_value_correct_when_known': int(value_correct.sum()),
            'present_margin_max': float(data['margins'][present].max()),
            'absent_margin_min': float(data['margins'][~present].min()),
            'first_false_known': [{'time': int(data['times'][i]), 'key': int(data['keys'][i]),
                                   'margin': float(data['margins'][i])}
                                  for i in torch.where(false_known)[0][:10]]}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Original individual mismatch')
    torch.set_num_threads(2)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    calibration = []
    for seed in CALIBRATION:
        calibration.extend(trajectory(model, seed))
        print(f'calibration {seed} complete', flush=True)
    max_present = max(float(item['margins'][item['present']].max()) for item in calibration)
    min_absent = min(float(item['margins'][~item['present']].min()) for item in calibration)
    threshold = (max_present + min_absent) / 2
    validation = []
    for seed in VALIDATION:
        validation.extend(trajectory(model, seed))
        print(f'validation {seed} complete', flush=True)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Original individual changed')
    report = {'individual_sha256': INDIVIDUAL_SHA, 'calibrated_threshold': threshold,
              'calibration_baseline': [summarize(item, 0.) for item in calibration],
              'calibration_threshold': [summarize(item, threshold) for item in calibration],
              'validation_baseline': [summarize(item, 0.) for item in validation],
              'validation_threshold': [summarize(item, threshold) for item in validation],
              'limitations': ['Same trained model and finite address basis for every trajectory.',
                              'Repeated checkpoints and object/time combinations are not independent samples.',
                              'Threshold is midpoint of calibration extremes; not a calibrated probability.',
                              'No unseen models, free-form language, provenance, or crash recovery tested.',
                              'Value correctness uses original argmax even when the adjusted threshold changes known status.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
