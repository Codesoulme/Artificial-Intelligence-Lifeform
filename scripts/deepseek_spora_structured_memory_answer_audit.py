"""Audit generation from a persisted structured packet without prompt injection."""
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


OUTPUT = ROOT / 'runs/deepseek-spora-structured-memory-answer-audit-v2-20261113'
BASELINE = ROOT / 'runs/deepseek-spora-retrained-natural-observation-transaction-20261113/working-individual.pt'
WORKING = OUTPUT / 'working-individual.pt'
CHECKPOINT = ROOT / 'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
DECODER = ROOT / 'runs/deepseek-spora-structured-memory-text-decoder-20261113/candidate.pt'
KEY = 16
IDENTITY = '小赵'
FACT = '苹果'


def open_chat():
    runtime.CHECKPOINT = CHECKPOINT
    runtime.WORKING_CHECKPOINT = WORKING
    return runtime.load_chat()


def write_phase():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not BASELINE.exists() or not DECODER.exists():
        raise FileNotFoundError('Structured answer audit inputs are missing')
    OUTPUT.mkdir(parents=False, exist_ok=False)
    shutil.copy2(BASELINE, WORKING)
    chat = open_chat()
    write = chat.observe_structured_text(KEY, IDENTITY, FACT)
    answer = chat.generate_structured_text(KEY, DECODER)
    result = {'phase': 'write', 'write': write, 'answer': answer,
              'working_sha256': runtime.sha(WORKING)}
    (OUTPUT / 'write.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': 'write', 'action': write['action'],
                      'answer': answer.get('answer')}, ensure_ascii=False), flush=True)


def read_phase():
    before = json.loads((OUTPUT / 'write.json').read_text(encoding='utf-8'))
    chat = open_chat()
    answer = chat.generate_structured_text(KEY, DECODER)
    text = answer.get('answer', '')
    result = {
        'phase': 'read-after-reload',
        'answer': answer,
        'working_sha256': runtime.sha(WORKING),
        'same_checkpoint': before['working_sha256'] == runtime.sha(WORKING),
        'same_answer': text == before['answer'].get('answer'),
        'identity_present': IDENTITY in text,
        'fact_present': FACT in text,
    }
    result['passed'] = all((before['write']['action'] == 'structured_text_written',
                            answer.get('accepted') is True, result['same_checkpoint'],
                            result['same_answer'], result['identity_present'],
                            result['fact_present']))
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': 'read-after-reload', 'passed': result['passed'],
                      'answer': text}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    write_phase() if args.phase == 'write' else read_phase()
