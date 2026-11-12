"""Repeat varied temporal runtime comparison using the isolated v6 intent head."""
import json

import torch

from scripts import deepseek_spora_temporal_regression_runtime as audit

SOURCE = audit.ROOT / 'runs/deepseek-spora-read-intent-gate-v6-20261112'
OUTPUT = audit.ROOT / 'runs/deepseek-spora-temporal-v5-varied-intent-v6-20261112.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    path = SOURCE / 'intent.pt'
    report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    digest = audit.runtime.sha(path)
    if digest != report['intent_sha256']:
        raise ValueError('Intent candidate digest mismatch')
    candidate = torch.load(path, map_location='cpu', weights_only=True)
    original_loader = audit.runtime.load_chat
    original_output = audit.OUTPUT

    def loader(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
            if candidate.get(key) != chat.intent_data.get(key):
                raise ValueError(f'Intent provenance mismatch: {key}')
        chat.intent_data = candidate
        return chat

    try:
        audit.runtime.load_chat = loader
        audit.CANDIDATE_DIR = audit.ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
        audit.QUESTIONS = (
            ('current', '目前{name}最喜欢哪种饮料？', 0, '水'),
            ('past', '更换喜好之前，{name}喜欢喝什么饮料？', 1, '咖啡'),
            ('later', '最后一次更新后，{name}喜欢什么饮品？', 2, '水'),
        )
        audit.OUTPUT = OUTPUT
        audit.main()
    finally:
        audit.runtime.load_chat = original_loader
        audit.OUTPUT = original_output
    if audit.runtime.sha(path) != digest:
        raise ValueError('Intent candidate changed')


if __name__ == '__main__':
    main()
