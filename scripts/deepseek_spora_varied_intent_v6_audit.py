"""Compare existing intent v6 on the varied temporary guarded runtime cases."""
import json

import torch

from scripts import deepseek_spora_candidate_guard_varied_audit as varied
from scripts import deepseek_spora_candidate_pre_read_guard_audit as boundary
from scripts import deepseek_spora_pre_read_guard_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime

CANDIDATE = audit.p.ROOT / 'runs/deepseek-spora-read-intent-gate-v6-20261112'
OUTPUT = audit.p.ROOT / 'runs/deepseek-spora-varied-intent-v6-audit-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    path = CANDIDATE / 'intent.pt'
    report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    digest = runtime.sha(path)
    if digest != report['intent_sha256']:
        raise ValueError('Intent candidate digest mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    original_loader, original_cases = runtime.load_chat, audit.CASES

    def load_with_intent(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
            if candidate.get(key) != chat.intent_data.get(key):
                raise ValueError(f'Intent provenance mismatch: {key}')
        chat.intent_data = candidate
        return chat

    try:
        runtime.load_chat = load_with_intent
        audit.CASES = varied.CASES
        boundary.main(OUTPUT)
    finally:
        runtime.load_chat, audit.CASES = original_loader, original_cases
    if runtime.sha(path) != digest:
        raise ValueError('Intent candidate changed')


if __name__ == '__main__':
    main()
