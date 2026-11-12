"""Verify candidate gates and per-instance floor across durable temporary reloads."""
import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.language_training import frozen_digest

ROOT = Path(__file__).resolve().parents[1]
INTENT_DIR = ROOT / 'runs/deepseek-spora-read-intent-gate-v7-20261112'
TEMPORAL_DIR = ROOT / 'runs/deepseek-spora-temporal-gate-v5-contrast-20261112'
OUTPUT = ROOT / 'runs/deepseek-spora-persistent-low-floor-chat-20261112.json'
KEY, NAME, _ = ENROLLMENTS[0]
QUESTIONS = (
    ('current', f'目前{NAME}最喜欢哪种饮料？', '水'),
    ('past', f'{NAME}以前喜欢哪种饮料？', '咖啡'),
    ('later', f'最后一次更新后，{NAME}喜欢什么饮品？', '水'),
)


def candidate_paths():
    intent_path = INTENT_DIR / 'intent.pt'
    temporal_path = TEMPORAL_DIR / 'temporal.pt'
    intent_report = json.loads((INTENT_DIR / 'result.json').read_text(encoding='utf-8'))
    temporal_report = json.loads((TEMPORAL_DIR / 'result.json').read_text(encoding='utf-8'))
    if runtime.sha(intent_path) != intent_report['intent_sha256'] or runtime.sha(temporal_path) != temporal_report['temporal_sha256']:
        raise ValueError('Candidate digest mismatch')
    return (torch.load(intent_path, map_location='cpu', weights_only=True),
            torch.load(temporal_path, map_location='cpu', weights_only=True),
            runtime.sha(intent_path), runtime.sha(temporal_path))


def install(chat, intent, temporal):
    for key in ('base_sha256', 'feature_variant_sha256', 'router_sha256', 'entity_gate_sha256'):
        if intent.get(key) != chat.intent_data.get(key):
            raise ValueError(f'Intent provenance mismatch: {key}')
    for key in ('base_sha256', 'feature_variant_sha256', 'classes'):
        if temporal.get(key) != chat.temporal_data.get(key):
            raise ValueError(f'Temporal provenance mismatch: {key}')
    temporal_head = nn.Linear(chat.temporal.in_features, 4).eval()
    temporal_head.load_state_dict(temporal['state'], strict=True)
    chat.intent_data = intent
    chat.temporal = temporal_head
    chat.temporal_data = temporal
    chat.read_intent_recovery_floor = -1.0


def ask(chat, phase):
    rows = []
    before = frozen_digest(chat.organism, chat.state)
    for label, question, expected in QUESTIONS:
        result = chat.answer(question)
        rows.append({'phase': phase, 'kind': label, 'question': question, 'expected': expected,
                     'answer': result.get('answer'), 'reason': result.get('reason'),
                     'accepted': result.get('accepted'), 'temporal_intent': result.get('temporal_intent'),
                     'key': result.get('key'), 'correct': result.get('answer') == expected})
    after = frozen_digest(chat.organism, chat.state)
    if before != after:
        raise ValueError('Read modified memory state')
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source_checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    intent, temporal, intent_sha, temporal_sha = candidate_paths()
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory) / 'individual.pt'
        old_work = runtime.WORKING_CHECKPOINT
        runtime.WORKING_CHECKPOINT = work
        try:
            chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
            install(chat, intent, temporal)
            initial_digest = frozen_digest(chat.organism, chat.state)
            decisions = []
            for value in (1, 6):
                decisions.append(chat.observe_sourced(KEY, value, 'direct', 1.0))
            after_writes_digest = frozen_digest(chat.organism, chat.state)
            first = ask(chat, 'before_reload')
            first_work_sha = runtime.sha(work)
            reloaded = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
            install(reloaded, intent, temporal)
            second = ask(reloaded, 'after_reload')
            second_work_sha = runtime.sha(work)
            if first_work_sha != second_work_sha:
                raise ValueError('Read changed durable checkpoint')
            if [r['answer'] for r in first] != [r['answer'] for r in second]:
                raise ValueError('Reload changed answers')
            if frozen_digest(reloaded.organism, reloaded.state) != after_writes_digest:
                raise ValueError('Reloaded state digest differs after reads')
            result = {'decisions': decisions, 'initial_digest': initial_digest,
                      'after_writes_digest': after_writes_digest, 'before_reload': first,
                      'after_reload': second, 'working_checkpoint_sha256': second_work_sha,
                      'source_checkpoint_sha256': source_checkpoint_sha, 'base_sha256': runtime.WEIGHT_SHA,
                      'intent_sha256': intent_sha, 'temporal_sha256': temporal_sha,
                      'limitations': ['Temporary controlled numeric writes and one known name.',
                                      'Candidate gates and floor are explicitly installed only in memory.',
                                      'No natural-language write, unknown name, or open-world answer evaluated.',
                                      'Temporary working checkpoint deleted after the run; default individual untouched.']}
        finally:
            runtime.WORKING_CHECKPOINT = old_work
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before_reload': first, 'after_reload': second}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
