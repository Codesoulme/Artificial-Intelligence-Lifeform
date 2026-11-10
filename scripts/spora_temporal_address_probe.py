"""Check native old/current values and derive their explicit temporal addresses."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest

OUTPUT = ROOT / 'runs/spora-temporal-address-probe-final-20261110.json'


@torch.no_grad()
def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected individual mismatch')
    torch.set_num_threads(2)
    model, original, _ = __import__('spora.growing', fromlist=['load']).load(INDIVIDUAL, 'cpu')
    before = frozen_digest(model, original)
    state = fresh_controlled_state(model)
    key, old_value, current_value = 8, 1, 6
    state, first = observe_sourced(model, state, key, old_value, 'direct', 1.0)
    state, second = observe_sourced(model, state, key, current_value, 'direct', 1.0)
    device = next(model.parameters()).device
    outputs = {}
    addresses = {}
    for label, mode, time, expected in (('current', 0, 0, current_value),
                                        ('old_event', 1, 0, old_value),
                                        ('new_event', 1, 1, current_value)):
        k = torch.tensor([key], dtype=torch.long, device=device)
        m = torch.tensor([mode], dtype=torch.long, device=device)
        t = torch.tensor([time], dtype=torch.long, device=device)
        tokens, bound = model.representation(k, m, t)
        q = [model.memory_address(site, bound, k, m) for site in model.memories]
        logits = query(model, state, k, m, t)
        outputs[label] = {'prediction': int(logits.argmax(-1)), 'expected': expected,
                          'correct': int(logits.argmax(-1)) == expected, 'mode': mode, 'event_time': time}
        addresses[label] = [item[0].tolist() for item in q]
        address_norms = [float(item[0].norm()) for item in q]
        outputs[label]['address_norms'] = address_norms
    if frozen_digest(model, original) != before or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected source changed')
    report = {'status': 'temporal_address_development_probe', 'individual_sha256': INDIVIDUAL_SHA,
              'key': key, 'writes': [first, second], 'cursor': state.cursor, 'page_count': len(state.pages),
              'outputs': outputs, 'addresses': addresses,
              'limitations': ['Two controlled direct writes to one address and one active history page.',
                              'Native SPORA query only; does not test the DeepSeek reader or natural-language time parsing.',
                              'The timestamp is explicitly supplied by the probe, not inferred by a model.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'outputs': outputs}), flush=True)


if __name__ == '__main__':
    run()
