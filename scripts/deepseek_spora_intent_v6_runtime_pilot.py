"""Run the candidate through the earlier temporary runtime contrast."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_intent_end_to_end_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime

CANDIDATE = ROOT / 'runs/deepseek-spora-read-intent-gate-v6-20261112'
audit.OUTPUT = ROOT / 'runs/deepseek-spora-intent-v6-runtime-pilot-20261112.json'
original_load_chat = runtime.load_chat


def load_with_candidate(*args, **kwargs):
    info = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    path = CANDIDATE / 'intent.pt'
    if runtime.sha(path) != info['intent_sha256']:
        raise ValueError('Candidate digest mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    chat = original_load_chat(*args, **kwargs)
    for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
        if candidate.get(key) != chat.intent_data.get(key):
            raise ValueError(f'Candidate provenance mismatch: {key}')
    chat.intent_data = candidate
    return chat


if __name__ == '__main__':
    with patch.object(runtime, 'load_chat', side_effect=load_with_candidate):
        audit.main()
