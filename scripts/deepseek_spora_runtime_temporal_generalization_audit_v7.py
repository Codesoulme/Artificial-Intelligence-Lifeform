"""Compare the v7 read-intent head in the current runtime audit."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_runtime_chat as runtime
from scripts import deepseek_spora_runtime_temporal_generalization_audit as audit


CANDIDATE = ROOT / 'runs/deepseek-spora-read-intent-gate-v7-20261112/intent.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-runtime-temporal-generalization-audit-v7-20261113.json'


def main():
    candidate = torch.load(CANDIDATE, map_location='cpu', weights_only=True)
    original_loader = runtime.load_chat

    def loader(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
            if candidate.get(key) != chat.intent_data.get(key):
                raise ValueError(f'Intent provenance mismatch: {key}')
        chat.intent_data = candidate
        return chat

    original_output = audit.OUTPUT
    try:
        runtime.load_chat = loader
        audit.OUTPUT = OUTPUT
        audit.main()
    finally:
        runtime.load_chat = original_loader
        audit.OUTPUT = original_output


if __name__ == '__main__':
    main()
