"""Measure finite native memory retention and state size across growing histories."""
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load, query, scalars
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-capacity-history-negative-20261109.json'
CHECKPOINTS = (0, 8, 16, 32, 48, 64, 80, 96, 128, 192)
SEEDS = (4615, 4616, 4617)


@torch.no_grad()
def assess(model, state, expected, events):
    device = next(model.parameters()).device
    keys = torch.arange(model.config.key_count, dtype=torch.long, device=device)
    zeros = torch.zeros_like(keys)
    predicted = query(model, state, keys, zeros, zeros).argmax(-1).tolist()
    rows = []
    for key, value in enumerate(predicted):
        truth = expected.get(key, model.config.unknown)
        rows.append({'key': key, 'expected': truth, 'predicted': value})
    known = [row for row in rows if row['expected'] != model.config.unknown]
    unknown = [row for row in rows if row['expected'] == model.config.unknown]
    history_rows = []
    absent_rows = []
    if events:
        history_keys = torch.tensor([event['key'] for event in events], dtype=torch.long, device=device)
        times = torch.tensor([event['step'] - 1 for event in events], dtype=torch.long, device=device)
        history_predicted = query(model, state, history_keys, torch.ones_like(history_keys), times).argmax(-1).tolist()
        history_rows = [{'step': event['step'], 'key': event['key'], 'expected': event['value'], 'predicted': value}
                        for event, value in zip(events, history_predicted)]
        # Each global event records exactly one key; a different key is absent at that time.
        absent_keys = (history_keys + 1) % model.config.key_count
        absent_predicted = query(model, state, absent_keys, torch.ones_like(absent_keys), times).argmax(-1).tolist()
        absent_rows = [{'time': event['step'] - 1, 'key': int(key), 'predicted': value}
                       for event, key, value in zip(events, absent_keys.tolist(), absent_predicted)]
    if state.cursor < capacity(model, state):
        future_keys = keys
        future_times = torch.full_like(keys, state.cursor)
        future_predicted = query(model, state, future_keys, torch.ones_like(keys), future_times).argmax(-1).tolist()
    else:
        future_predicted = []
    query(model, state, keys[:1], zeros[:1], zeros[:1])
    timings = []
    for key in range(8):
        started = time.perf_counter()
        query(model, state, keys[key:key + 1], zeros[key:key + 1], zeros[key:key + 1])
        timings.append((time.perf_counter() - started) * 1000)
    return {'cursor': state.cursor, 'pages': len(state.pages), 'capacity': capacity(model, state),
            'known': len(known), 'known_value_correct': sum(r['expected'] == r['predicted'] for r in known),
            'false_unknown': sum(r['predicted'] == model.config.unknown for r in known),
            'unknown': len(unknown), 'false_known': sum(r['predicted'] != model.config.unknown for r in unknown),
            'history_examples': len(history_rows),
            'history_correct': sum(r['expected'] == r['predicted'] for r in history_rows),
            'history_rows': history_rows,
            'absent_history_examples': len(absent_rows),
            'absent_history_false_known': sum(r['predicted'] != model.config.unknown for r in absent_rows),
            'absent_history_rows': absent_rows,
            'future_examples': len(future_predicted),
            'future_false_known': sum(value != model.config.unknown for value in future_predicted),
            'state_scalars': scalars(state), 'single_query_median_ms': statistics.median(timings),
            'single_query_samples_ms': timings, 'rows': rows}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Original individual mismatch')
    torch.set_num_threads(2)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    histories = []
    for seed in SEEDS:
        rng = random.Random(seed)
        state = fresh_controlled_state(model)
        expected, events, phases = {}, [], []
        order = list(range(model.config.key_count))
        rng.shuffle(order)
        phases.append(assess(model, state, expected, events))
        for step in range(1, CHECKPOINTS[-1] + 1):
            if state.cursor >= capacity(model, state):
                state = grow(model, state)
            key = order[step - 1] if step <= len(order) else rng.choice(order)
            previous = expected.get(key)
            value = rng.randrange(model.config.unknown)
            if previous is not None and value == previous:
                value = (value + 1) % model.config.unknown
            state, decision = observe_sourced(model, state, key, value, 'direct', 1.0)
            expected[key] = value
            events.append({'step': step, 'key': key, 'value': value,
                           'action': decision['action'], 'current_write': decision['current_write']})
            if step in CHECKPOINTS:
                phases.append(assess(model, state, expected, events))
        histories.append({'seed': seed, 'events': events, 'phases': phases})
        print(f'seed {seed} complete', flush=True)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Original individual changed')
    report = {'status': 'native_finite_memory_capacity_development_only', 'individual_sha256': INDIVIDUAL_SHA,
              'checkpoints': CHECKPOINTS, 'histories': histories,
              'limitations': ['Three random trajectories share the same model parameters and address basis.',
                              'Only 64 finite keys and 16 values, with trusted direct observations.',
                              'Reference values are recorded solely for evaluation, not supplied to model queries.',
                              'State scalar counts exclude model weights, serialization overhead and filesystem metadata.',
                              'Single-query CPU timings are small local samples, not a production latency benchmark.',
                              'Negative history uses one neighboring key per event and the next unfilled time when allocated.',
                              'No language-model generation, independent individuals, or crash recovery is tested.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([{'seed': item['seed'], 'phases': [
        {k: v for k, v in phase.items() if k not in ('rows', 'history_rows', 'absent_history_rows', 'single_query_samples_ms')}
        for phase in item['phases']]} for item in histories]), flush=True)


if __name__ == '__main__':
    run()
