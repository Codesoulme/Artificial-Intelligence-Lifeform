"""Matched crossed-value pilot with oracle-selected write positions."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_ttt_selective_write_audit import selected_memory

OUT = ROOT / 'runs/deepseek-spora-ttt-selective-crossed-20261114'


def build_selected_case(model, tokenizer, wrappers, name, value, question_template=None):
    fact = f'{name}喜欢喝{value}。'
    memory, positions = selected_memory(model, tokenizer, wrappers, fact, value)
    if not positions:
        raise ValueError('Selected memory must contain at least one write')
    return {'memory': {layer: tensor.detach().cpu().half() for layer, tensor in memory.items()}}


if __name__ == '__main__':
    crossed.OUT = OUT
    pilot.build_case = build_selected_case
    crossed.main()
    path = OUT / 'result.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    result['status'] = 'oracle_selected_crossed_value_development_pilot'
    result['write_mode'] = 'oracle_value_token_positions'
    result['limitations'].append('Value-token positions are selected using the known answer; not autonomous.')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
