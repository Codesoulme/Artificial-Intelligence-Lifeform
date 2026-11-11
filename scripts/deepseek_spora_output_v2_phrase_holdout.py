"""Paired adapter comparison on held names and newly authored question forms."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))
import torch
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.deepseek_spora_runtime_chat import load_chat
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.memory_output import MemoryOutputAdapter

SOURCE = ROOT/'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-output-v2-phrase-holdout-20261111.json'
TEMPLATES = ('从先前保存的资料来看，{name}当下想喝哪种饮料？',
             '能否说出{name}目前所选择的饮品名称？',
             '有关{name}的饮料选择，记录的是哪一种？')
VALUES = (0, 2, 5, 7)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    if sha(SOURCE/'adapter.pt') != report['adapter_sha256']:
        raise ValueError('Candidate digest mismatch')
    payload = torch.load(SOURCE/'adapter.pt', map_location='cpu', weights_only=True)
    if payload.get('base_sha256') != WEIGHT_SHA:
        raise ValueError('Candidate base mismatch')
    torch.set_num_threads(2)
    chat = load_chat()
    old = chat.decoder.output_adapter
    new = MemoryOutputAdapter(**payload['config']).to('cuda').eval()
    new.load_state_dict(payload['state'], strict=True)
    rows = []
    for index in (6, 7):
        key, name = KEYS[index], NAMES[index]
        for template in TEMPLATES:
            question = template.format(name=name)
            prompt = closed_think_inputs(chat.tokenizer, question, 'cuda')
            for value in VALUES:
                state = fresh_controlled_state(chat.organism)
                state, decision = observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
                if not decision['current_write']:
                    raise ValueError('Controlled write rejected')
                keys = torch.tensor([key], device='cuda')
                zeros = torch.zeros_like(keys)
                logits = query(chat.organism, state, keys, zeros, zeros)
                args = _memory_args(chat.organism, state, {'key': key, 'intent': 'current'})
                for label, adapter in (('old', old), ('candidate', new)):
                    chat.decoder.output_adapter = adapter
                    with torch.inference_mode():
                        ids = chat.decoder.generate(**prompt, **args, memory_output_logits=logits,
                                     max_new_tokens=12, do_sample=False,
                                     pad_token_id=chat.tokenizer.pad_token_id, eos_token_id=chat.tokenizer.eos_token_id)
                    answer = chat.tokenizer.decode(ids[0, prompt['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    rows.append({'question': question, 'value': value, 'adapter': label,
                                 'expected': WORDS[value], 'native_value': int(logits.argmax(-1)[0]),
                                 'answer': answer, 'exact': answer in (WORDS[value], WORDS[value]+'.', WORDS[value]+'。')})
            print(json.dumps({'name_index': index, 'template_index': TEMPLATES.index(template)}), flush=True)
    chat.decoder.output_adapter = old
    result = {'candidate_sha256': sha(SOURCE/'adapter.pt'), 'rows': rows,
              'old_exact': sum(r['exact'] for r in rows if r['adapter'] == 'old'),
              'candidate_exact': sum(r['exact'] for r in rows if r['adapter'] == 'candidate'),
              'cases_per_adapter': len(rows)//2,
              'native_correct': sum(r['native_value'] == r['value'] for r in rows[::2]),
              'limitations': ['Two names excluded from output-head training but known to the base and router.',
                              'Three authored templates and four finite value classes; candidate saw all value classes in training.',
                              'Addresses supplied directly; no full language routing or unknown memory tested.',
                              'Exact match can penalize correct longer or differently formatted answers.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('old_exact', 'candidate_exact', 'cases_per_adapter', 'native_correct')}), flush=True)


if __name__ == '__main__':
    run()
