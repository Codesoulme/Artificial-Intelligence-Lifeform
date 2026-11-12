"""Run the v7 read-intent head in the same temporary end-to-end audit."""
import json
import torch

from scripts import deepseek_spora_candidate_guard_varied_audit as varied
from scripts import deepseek_spora_pre_read_guard_audit as audit
from scripts import deepseek_spora_runtime_chat as runtime

CANDIDATE = audit.p.ROOT / 'runs/deepseek-spora-read-intent-gate-v7-20261112'
OUTPUT = audit.p.ROOT / 'runs/deepseek-spora-varied-intent-v7-audit-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    path = CANDIDATE / 'intent.pt'
    report = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    digest = runtime.sha(path)
    if digest != report['intent_sha256']:
        raise ValueError('Intent candidate digest mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    original_loader, original_cases, original_output = runtime.load_chat, audit.CASES, audit.OUTPUT

    def loader(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
            if candidate.get(key) != chat.intent_data.get(key):
                raise ValueError(f'Intent provenance mismatch: {key}')
        chat.intent_data = candidate
        return chat

    try:
        runtime.load_chat = loader
        audit.CASES = varied.CASES
        audit.OUTPUT = OUTPUT
        audit.main()
    finally:
        runtime.load_chat, audit.CASES, audit.OUTPUT = original_loader, original_cases, original_output
    if runtime.sha(path) != digest:
        raise ValueError('Intent candidate changed')


if __name__ == '__main__':
    main()
