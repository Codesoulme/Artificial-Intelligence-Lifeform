"""Stress the exploratory provenance scanner over separate randomized trajectories."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_provenance_scan_audit import audit
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import capacity, grow, load
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-provenance-random-audit-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected individual mismatch')
    torch.set_num_threads(2)
    model, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    trials = []
    for seed in (8203, 8209, 8219):
        rng = random.Random(seed)
        state = fresh_controlled_state(model)
        writes = []
        for _ in range(96):
            if state.cursor == capacity(model, state):
                state = grow(model, state)
            key = rng.randrange(16)
            value = rng.randrange(model.config.unknown)
            source = rng.choices(('direct', 'report', 'inference'), weights=(7, 2, 1))[0]
            state, decision = observe_sourced(model, state, key, value, source, 1.0)
            writes.append({'key': key, 'value': value, 'source': source, **decision})
        # All active addresses plus two absent ones, not chosen using the outcome.
        result = audit(model, state, writes, list(range(18)))
        trials.append({'seed': seed, 'writes': writes, 'result': result})
        print(json.dumps({'seed': seed, **{k: v for k, v in result.items() if k not in ('rows', 'selection')}}), flush=True)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or frozen_digest(model, original) != before:
        raise ValueError('Protected individual changed')
    report = {'status': 'randomized_finite_address_stress_only', 'individual_sha256': INDIVIDUAL_SHA,
              'trials': trials, 'limitations': [
                  'Three pseudorandom synthetic trajectories use the same trained individual.',
                  'Sources and values are artificial; no natural language or DeepSeek inference.',
                  'Oracle treats actual controller decisions as correct, including deferred writes.',
                  'Scanning all times scales linearly; no index or asynchronous retrieval is measured.',
                  'No reload, sealing, wall-clock semantics or open-world confidence guarantee.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
