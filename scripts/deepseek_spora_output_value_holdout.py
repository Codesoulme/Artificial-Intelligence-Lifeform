"""Test output adapter on values never used to train it, without changing state."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.deepseek_spora_runtime_chat import load_chat
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced

OUTPUT = ROOT/'runs/deepseek-spora-output-value-holdout-20261111.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    chat = load_chat()
    key = KEYS[0]
    prompt = closed_think_inputs(chat.tokenizer, '小林现在喜欢喝什么？', 'cuda')
    rows = []
    for value in range(len(WORDS)):
        state = fresh_controlled_state(chat.organism)
        state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Controlled write rejected')
        keys = torch.tensor([key], device='cuda')
        zeros = torch.zeros_like(keys)
        logits = query(chat.organism, state, keys, zeros, zeros)
        args = _memory_args(chat.organism, state, {'key': key, 'intent': 'current'})
        with torch.inference_mode():
            ids = chat.decoder.generate(**prompt, **args, memory_output_logits=logits,
                         max_new_tokens=8, do_sample=False, pad_token_id=chat.tokenizer.pad_token_id,
                         eos_token_id=chat.tokenizer.eos_token_id)
        answer = chat.tokenizer.decode(ids[0, prompt['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        rows.append({'value': value, 'trained_value': value in (1, 6), 'expected': WORDS[value],
                     'native_value': int(logits.argmax(-1)[0]), 'answer': answer,
                     'exact': answer in (WORDS[value], WORDS[value]+'.', WORDS[value]+'。')})
    report = {'rows': rows, 'trained_total': 2, 'heldout_total': 6,
              'trained_exact': sum(r['exact'] for r in rows if r['trained_value']),
              'heldout_exact': sum(r['exact'] for r in rows if not r['trained_value']),
              'limitations': ['One question and one supplied entity address.',
                              'Synthetic controlled writes; native prediction checked independently.',
                              'No natural-language routing, open chat or persistence tested.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('trained_total', 'trained_exact', 'heldout_total', 'heldout_exact')}))


if __name__ == '__main__':
    run()
