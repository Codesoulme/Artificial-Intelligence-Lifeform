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


OUTPUT = ROOT / 'runs/deepseek-spora-structured-write-gate-audit-v2-20261113'
BASELINE = ROOT / 'runs/deepseek-spora-retrained-natural-observation-transaction-20261113/working-individual.pt'
WORKING = OUTPUT / 'working-individual.pt'
CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
KEY = 16
IDENTITY = '小梅'
FACT = '喜欢喝茶'
FINAL_FACT = '喜欢喝咖啡'


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
    first = chat.observe_structured_text(KEY, IDENTITY, FACT)
    repeat = chat.observe_structured_text(KEY, IDENTITY, FACT, query_text='小梅喜欢喝什么？')
    unrelated = chat.observe_structured_text(KEY, IDENTITY, '喜欢读书', query_text='天气怎么样？')
    correction = chat.observe_structured_text(KEY, IDENTITY, '喜欢喝咖啡', query_text='小梅喜欢喝什么？')
    record = chat.read_structured_text(KEY)
    result = {'phase': 'write', 'first': first, 'repeat': repeat,
              'unrelated': unrelated, 'correction': correction, 'record': record,
              'working_sha256': runtime.sha(WORKING)}
    (OUTPUT / 'write.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': 'write', 'revision': record['revision'],
                      'actions': [first['action'], repeat['action'], unrelated['action'], correction['action']]},
                     ensure_ascii=False), flush=True)


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
        'fact_correct': record is not None and record['fact_text'] == FINAL_FACT,
    }
    result['repeat_deferred'] = before['repeat']['action'] == 'structured_text_deferred'
    result['unrelated_deferred'] = before['unrelated']['action'] == 'structured_text_deferred'
    result['correction_written'] = before['correction']['action'] == 'structured_text_written'
    result['passed'] = all((result['same_checkpoint'], result['identity_correct'],
                            result['fact_correct'], result['repeat_deferred'],
                            result['unrelated_deferred'], result['correction_written']))
    (OUTPUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'phase': 'read-after-reload', 'passed': result['passed']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('write', 'read'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    write_phase() if args.phase == 'write' else read_phase()
