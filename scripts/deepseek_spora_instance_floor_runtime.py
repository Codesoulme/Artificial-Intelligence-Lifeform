"""End-to-end runtime test using the explicit per-instance floor option."""
import json
from unittest.mock import patch

import torch

from scripts import deepseek_spora_temporal_regression_runtime as audit

ROOT = audit.ROOT
INTENT_DIR = ROOT / 'runs/deepseek-spora-read-intent-gate-v7-20261112'
TEMPORAL_DIR = ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
OUTPUT = ROOT / 'runs/deepseek-spora-instance-floor-runtime-20261112.json'
QUESTIONS = (
    ('current', '目前{name}最喜欢哪种饮料？', 0, '水'),
    ('past', '更换喜好之前，{name}喜欢喝什么饮料？', 1, '咖啡'),
    ('later', '最后一次更新后，{name}喜欢什么饮品？', 2, '水'),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    intent_path = INTENT_DIR / 'intent.pt'
    intent_report = json.loads((INTENT_DIR / 'result.json').read_text(encoding='utf-8'))
    temporal_path = TEMPORAL_DIR / 'temporal.pt'
    temporal_report = json.loads((TEMPORAL_DIR / 'result.json').read_text(encoding='utf-8'))
    if audit.runtime.sha(intent_path) != intent_report['intent_sha256'] or audit.runtime.sha(temporal_path) != temporal_report['temporal_sha256']:
        raise ValueError('Candidate digest mismatch')
    intent = torch.load(intent_path, map_location='cpu', weights_only=True)
    source_sha = audit.runtime.sha(audit.runtime.CHECKPOINT)
    original_loader = audit.runtime.load_chat
    original_output, original_questions = audit.OUTPUT, audit.QUESTIONS

    def loader(*args, **kwargs):
        chat = original_loader(*args, **kwargs)
        for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
            if intent.get(key) != chat.intent_data.get(key):
                raise ValueError(f'Intent provenance mismatch: {key}')
        chat.intent_data = intent
        chat.read_intent_recovery_floor = -1.0
        return chat

    try:
        audit.runtime.load_chat = loader
        audit.OUTPUT = OUTPUT
        audit.QUESTIONS = QUESTIONS
        audit.CANDIDATE_DIR = TEMPORAL_DIR
        audit.main()
    finally:
        audit.runtime.load_chat = original_loader
        audit.OUTPUT, audit.QUESTIONS = original_output, original_questions
    if audit.runtime.sha(intent_path) != intent_report['intent_sha256'] or audit.runtime.sha(audit.runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected candidate or checkpoint changed')


if __name__ == '__main__':
    main()
