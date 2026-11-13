"""Audit structured arbitrary-text memory through the main SporaChat runtime."""
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


OUTPUT = ROOT / 'runs/deepseek-spora-structured-text-read-after-write-20261113'
BASELINE = ROOT / 'runs/deepseek-spora-retrained-natural-observation-transaction-20261113/working-individual.pt'
WORKING = OUTPUT / 'working-individual.pt'
CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
KEY = 16
IDENTITY = '小梅'
FACT = '喜欢喝茶'


def open_chat():
    runtime.CHECKPOINT = CHECKPOINT
    runtime.WORKING_CHECKPOINT = WORKING
    return runtime.load_chat()


def write_phase():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    shutil.copy2(BASELINE, WORKING)
    chat = open_chat()
    write_result = chat.observe_structured_text(KEY, IDENTITY, FACT)
    record = chat.read_structured_text(KEY)
    result = {'phase': 'write', 'write': write_result, 'record': record,
              'working_sha256': runtime.sha(WORKING)}
    (OUTPUT / 'write.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': 'write', 'revision': record['revision']}, ensure_ascii=False), flush=True)


def read_phase():
    before = json.loads((OUTPUT / 'write.json').read_text(encoding='utf-8'))
    chat = open_chat()
    record = chat.read_structured_text(KEY)
    result = {
        'phase': 'read-after-reload',
        'record': record,
        'working_sha256': runtime.sha(WORKING),
        'same_checkpoint': before['working_sha256'] == runtime.sha(WORKING),
        'identity_correct': record is not None and record['identity_text'] == IDENTITY,
        'fact_correct': record is not None and record['fact_text'] == FACT,
    }
    result['passed'] = all((result['same_checkpoint'], result['identity_correct'], result['fact_correct']))
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': 'read-after-reload', 'passed': result['passed']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    write_phase() if args.phase == 'write' else read_phase()
