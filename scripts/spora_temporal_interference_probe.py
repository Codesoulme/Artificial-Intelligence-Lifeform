"""Measure old-event retrieval after later writes on a real SPORA address basis."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow, load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-temporal-interference-v2-20261111.json'
GAPS = (1, 2, 4, 8, 16, 32, 63, 64, 127)


def _address(model, key, event_time):
    keys = torch.tensor([key], dtype=torch.long)
    modes = torch.ones_like(keys)
    times = torch.tensor([event_time], dtype=torch.long)
    _, bound = model.representation(keys, modes, times % model.config.time_count)
    return model.memory_address(model.memories[-1], bound, keys, modes)


def _read(model, state, key, event_time):
    page = event_time // model.config.time_count
    matrix = state.pages[page][-1][0]
    return model.memories[-1].retrieve([matrix], _address(model, key, event_time))


def _target(model, value):
    token = torch.tensor([value + model.config.value_start], dtype=torch.long)
    return model.memories[-1].target(model.embedding(token))


def _top1(model, retrieved):
    values = torch.arange(model.config.unknown, dtype=torch.long) + model.config.value_start
    targets = model.memories[-1].target(model.embedding(values))
    scores = F.normalize(retrieved, dim=-1) @ F.normalize(targets, dim=-1).T
    return int(scores.argmax(-1).item())


def _append(model, state, key, value):
    if state.cursor >= len(state.pages) * model.config.time_count:
        state = grow(model, state)
    return observe_sourced(model, state, key, value, 'direct', 1.0)


def _trial(model, gap, pattern, trial):
    key = trial
    state = fresh_controlled_state(model)
    state, decision = _append(model, state, key, 1)
    if decision['action'] != 'new':
        raise ValueError(f'Initial write was not accepted: {decision}')
    original_time = 0
    old_before = _read(model, state, key, original_time)
    filler_actions = []
    last_value = 1
    for index in range(gap):
        filler_key = key if pattern == 'same_key' else (key + index + 1) % model.config.key_count
        value = 2 + ((index + trial) % (model.config.unknown - 2))
        state, filler = _append(model, state, filler_key, value)
        filler_actions.append(filler['action'])
        if filler_key == key and filler['current_write']:
            last_value = value
    correction_time = state.cursor
    correction_value = 6 if last_value != 6 else 7
    state, correction = _append(model, state, key, correction_value)
    if correction['action'] != 'correct':
        raise ValueError(f'Correction was not accepted: {correction}')
    old_after = _read(model, state, key, original_time)
    new_after = _read(model, state, key, correction_time)
    old_target = _target(model, 1)
    new_target = _target(model, correction_value)
    return {
        'trial': trial,
        'pattern': pattern,
        'gap': gap,
        'correction_time': correction_time,
        'correction_value': correction_value,
        'correction_page': correction_time // model.config.time_count,
        'filler_deferred': sum(action == 'defer' for action in filler_actions),
        'old_target_cosine_before': float(F.cosine_similarity(old_before, old_target).item()),
        'old_target_cosine_after': float(F.cosine_similarity(old_after, old_target).item()),
        'old_value_top1_after': _top1(model, old_after),
        'old_value_top1_correct': _top1(model, old_after) == 1,
        'old_vector_relative_drift': float((old_after - old_before).norm() / old_before.norm().clamp_min(1e-8)),
        'new_target_cosine': float(F.cosine_similarity(new_after, new_target).item()),
        'new_value_top1_after': _top1(model, new_after),
        'new_value_top1_correct': _top1(model, new_after) == correction_value,
    }


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Original SPORA checkpoint digest mismatch')
    torch.manual_seed(81431)
    torch.set_num_threads(2)
    model, initial, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, initial)
    rows = [_trial(model, gap, pattern, trial)
            for trial, pattern in enumerate(('distinct_keys', 'same_key'))
            for gap in GAPS]
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, initial) != before:
        raise ValueError('Protected organism changed during interference test')
    report = {
        'status': 'real_individual_temporal_interference_development_probe',
        'individual_sha256': INDIVIDUAL_SHA,
        'individual_frozen_digest': before,
        'time_count': model.config.time_count,
        'memory_width': model.config.memory_width,
        'rows': rows,
        'summary': {
            'examples': len(rows),
            'worst_old_target_cosine_after': min(row['old_target_cosine_after'] for row in rows),
            'worst_old_vector_relative_drift': max(row['old_vector_relative_drift'] for row in rows),
            'worst_new_target_cosine': min(row['new_target_cosine'] for row in rows),
            'old_value_top1_correct': sum(row['old_value_top1_correct'] for row in rows),
            'new_value_top1_correct': sum(row['new_value_top1_correct'] for row in rows),
        },
        'limitations': [
            'Two hand-designed write sequences and one deterministic controller trajectory.',
            'Measures native memory vectors, not language-model answers or semantic intent routing.',
            'No concurrent writes, sleep consolidation, ambiguous facts or noisy sources tested.',
            'High target-vector cosine does not establish correct natural-language recall.',
        ],
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
