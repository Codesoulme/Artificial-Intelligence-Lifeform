"""Audit a natural-language write followed by a separate-process read."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_spora_runtime_chat as runtime


OUTPUT = ROOT / 'runs/deepseek-spora-read-after-write-audit-v2-20261113'
BASELINE = ROOT / 'runs/deepseek-spora-retrained-natural-observation-transaction-20261113/working-individual.pt'
WORKING = OUTPUT / 'working-individual.pt'
HEADS = ROOT / 'runs/deepseek-spora-natural-write-coverage-retrain-20261113/heads.pt'
CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
STATEMENT = '小梅偏爱的饮料是茶。'
QUESTION = '小梅现在喜欢喝什么饮料？'


def open_chat():
    runtime.CHECKPOINT = CHECKPOINT
    runtime.WORKING_CHECKPOINT = WORKING
    return runtime.load_chat(natural_write_heads_path=HEADS)


def write_phase():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    shutil.copy2(BASELINE, WORKING)
    chat = open_chat()
    observation = chat.observe_text(STATEMENT)
    answer = chat.answer(QUESTION)
    result = {
        'phase': 'write',
        'observation_action': observation['action'],
        'answer': answer.get('answer'),
        'accepted': answer.get('accepted'),
        'working_sha256': runtime.sha(WORKING),
    }
    (OUTPUT / 'write.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


def read_phase():
    before = json.loads((OUTPUT / 'write.json').read_text(encoding='utf-8'))
    chat = open_chat()
    answer = chat.answer(QUESTION)
    result = {
        'phase': 'read-after-reload',
        'answer': answer.get('answer'),
        'accepted': answer.get('accepted'),
        'working_sha256': runtime.sha(WORKING),
        'same_checkpoint': before['working_sha256'] == runtime.sha(WORKING),
        'same_answer': answer.get('answer') == before['answer'],
        'passed': (before['observation_action'] in ('new', 'correct')
                   and before['accepted'] is True
                   and answer.get('accepted') is True
                   and answer.get('answer') == '茶'
                   and answer.get('answer') == before['answer']
                   and before['working_sha256'] == runtime.sha(WORKING)),
    }
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    write_phase() if args.phase == 'write' else read_phase()
