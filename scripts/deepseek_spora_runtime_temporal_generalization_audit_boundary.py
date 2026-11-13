"""Compare the boundary locator candidate in the current runtime audit."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepseek_spora_runtime_chat as runtime
from scripts import deepseek_spora_runtime_temporal_generalization_audit as audit


CANDIDATE = ROOT / 'runs/deepseek-spora-boundary-train-pilot-20261112/locator.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-runtime-temporal-generalization-audit-boundary-20261113.json'


def main():
    candidate = torch.load(CANDIDATE, map_location='cpu', weights_only=True)
    original_loader = runtime.load_chat

    def loader(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        if candidate.get('feature_variant_sha256') != chat.router.get('feature_variant_sha256'):
            raise ValueError('Boundary locator feature provenance mismatch')
        chat.router['locator_state'] = candidate['state']
        chat.router['locator_threshold'] = candidate['threshold']
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
