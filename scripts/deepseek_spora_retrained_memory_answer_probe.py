"""Probe generated answers after an isolated retrained natural-memory write."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_spora_runtime_chat as runtime


WORKING = ROOT / 'runs/deepseek-spora-retrained-natural-observation-transaction-20261113/working-individual.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-retrained-memory-answer-probe-v3-20261113.json'
QUESTIONS = (
    '小梅现在喜欢喝什么饮料？',
    '小梅目前最喜欢哪种饮料？',
    '小梅偏爱的饮料是什么？',
    '小梅以前喜欢喝什么饮料？',
    '小梅更新后喜欢喝什么饮料？',
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not WORKING.exists():
        raise FileNotFoundError(WORKING)
    runtime.WORKING_CHECKPOINT = WORKING
    chat = runtime.load_chat()
    rows = []
    for question in QUESTIONS:
        result = chat.answer(question)
        rows.append({'question': question, **result})
    report = {
        'status': 'isolated_retrained_memory_answer_probe',
        'working_sha256': runtime.sha(WORKING),
        'rows': rows,
        'coffee_mentions': sum('咖啡' in (row.get('answer') or '') for row in rows),
        'limitations': [
            'Temporary isolated individual only.',
            'Three authored questions and finite beverage fact.',
            'Answer route still uses the existing response adapter; this probe does not prove open chat.',
        ],
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'accepted': [row['accepted'] for row in rows],
                      'answers': [row.get('answer') for row in rows],
                      'coffee_mentions': report['coffee_mentions']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
