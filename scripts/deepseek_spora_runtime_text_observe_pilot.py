"""Exercise SporaChat.observe_text on an isolated working individual."""
import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_spora_runtime_chat as runtime
from spora.growing import query


OUTPUT = ROOT / 'runs/deepseek-spora-runtime-text-observe-pilot-20261113'
WORKING = OUTPUT / 'working-individual.pt'
HEADS = ROOT / 'runs/deepseek-spora-natural-write-coverage-retrain-20261113/heads.pt'
CHECKPOINT = ROOT / 'runs/deepseek-spora-output-v2-checkpoint-20261111/individual-with-output-v2.pt'
CASES = (
    ('new', '小梅偏爱的饮料是茶。'),
    ('defer_negative', '小梅不喜欢喝茶。'),
    ('defer_unknown', '小梅偏爱的饮料是热巧克力。'),
    ('correct', '小梅偏爱的饮料是咖啡。'),
)


def open_chat():
    runtime.CHECKPOINT = CHECKPOINT
    runtime.WORKING_CHECKPOINT = WORKING
    return runtime.load_chat(natural_write_heads_path=HEADS)


def current_value(chat, key=16):
    device = next(chat.organism.parameters()).device
    return int(query(chat.organism, chat.state,
                     torch.tensor([key], device=device),
                     torch.zeros(1, dtype=torch.long, device=device),
                     torch.zeros(1, dtype=torch.long, device=device)).argmax(-1)[0])


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    chat = open_chat()
    rows = []
    for expected, text in CASES:
        result = chat.observe_text(text)
        rows.append({'expected_action': expected, 'text': text, **result,
                     'cursor_after': chat.state.cursor})
    pre = {'status': 'awaiting_reload', 'rows': rows,
           'before_reload': {'cursor': chat.state.cursor, 'value': current_value(chat)},
           'working_sha256': runtime.sha(WORKING)}
    (OUTPUT / 'pre-reload.json').write_text(json.dumps(pre, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    del chat
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({'actions': [row['action'] for row in rows]}, ensure_ascii=False), flush=True)


def finalize():
    pre = json.loads((OUTPUT / 'pre-reload.json').read_text(encoding='utf-8'))
    chat = open_chat()
    after = {'cursor': chat.state.cursor, 'value': current_value(chat),
             'working_sha256': runtime.sha(WORKING)}
    report = {'status': 'runtime_text_observe_isolated_pilot', 'rows': pre['rows'],
              'before_reload': pre['before_reload'], 'after_reload': after,
              'same_value': pre['before_reload']['value'] == after['value'],
              'same_checkpoint': pre['working_sha256'] == after['working_sha256'],
              'limitations': ['Temporary isolated working individual.',
                              'Finite beverage vocabulary and research head bundle.',
                              'Default runtime remains opt-in.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'actions': [row['action'] for row in report['rows']],
                      'same_value': report['same_value'],
                      'same_checkpoint': report['same_checkpoint']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    finalize() if args.finalize else run()
