"""Apply the retrained heads to the isolated natural-route stress audit."""
import json
from pathlib import Path

import torch
from torch import nn

from scripts import deepseek_spora_natural_route_stress_audit as stress
from scripts import deepseek_spora_natural_observation_transaction_pilot as pilot
from scripts import deepseek_spora_end_to_end_candidate_audit as route


ROOT = Path(__file__).resolve().parents[1]
HEADS = ROOT / 'runs/deepseek-spora-natural-write-coverage-retrain-20261113/heads.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-natural-route-retrained-stress-audit-20261113.json'


def build_components(chat):
    tokenizer = chat.tokenizer
    base = chat.base
    organism = chat.organism
    locator_data = torch.load(route.LOCATOR / 'locator.pt', map_location='cpu', weights_only=True)
    binding_data = torch.load(route.BINDING / 'binding.pt', map_location='cpu', weights_only=True)
    value_data = torch.load(route.VALUE, map_location='cpu', weights_only=True)
    heads = torch.load(HEADS, map_location='cpu', weights_only=True)
    locator = nn.Linear(locator_data['state']['weight'].shape[1], 1).eval()
    locator.load_state_dict(locator_data['state'], strict=True)
    polarity = nn.Linear(base.config.hidden_size * 2, len(route.CLASSES)).eval()
    polarity.load_state_dict(heads['polarity'], strict=True)
    polarity_data = {'threshold': heads['polarity_threshold']}
    value = route.SpanHead(base.config.hidden_size).eval()
    value.load_state_dict(heads['value'], strict=True)
    value_data = dict(value_data, threshold=heads['value_threshold'])
    bases = torch.stack([torch.nn.functional.normalize(site.current_basis.detach().float().cpu(), dim=-1)
                         for site in organism.memories])
    return tokenizer, locator, binding_data, polarity, polarity_data, value, value_data, bases


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not HEADS.exists():
        raise FileNotFoundError(HEADS)
    stress.pilot.build_components = build_components
    stress.OUTPUT = OUTPUT
    stress.main()
    report = json.loads(OUTPUT.read_text(encoding='utf-8'))
    report['status'] = 'natural_route_retrained_stress_audit_no_writes'
    report['heads_path'] = str(HEADS)
    report['heads_sha256'] = stress.runtime.sha(HEADS)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
