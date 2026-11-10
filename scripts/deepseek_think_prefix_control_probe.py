"""Check whether forcing a closed reasoning prefix affects the original base."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

OUTPUT = ROOT/'runs/deepseek-think-prefix-control-probe-20261111.json'
QUESTIONS = ('我上午先去了邮局，下午又去了菜市场。上午去了哪里？',
             '小叶不喝咖啡，她喜欢喝温水。她喜欢什么？')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight hash mismatch')
    torch.manual_seed(58331)
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    results = []
    with torch.inference_mode():
        for question in QUESTIONS:
            formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                                       add_generation_prompt=True)
            if not formatted.rstrip().endswith('<think>'):
                raise ValueError('Unexpected chat template')
            for mode in ('natural', 'forced_closed'):
                prompt = formatted + ('</think>\n\n' if mode == 'forced_closed' else '')
                encoded = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
                generated = base.generate(**encoded, max_new_tokens=192, use_cache=True, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                tail = generated[0, encoded['input_ids'].shape[1]:]
                text = tokenizer.decode(tail, skip_special_tokens=True)
                reasoning_closed = '</think>' in text
                answer = text.split('</think>', 1)[-1].strip() if (mode == 'forced_closed' or reasoning_closed) else ''
                row = {'question': question, 'mode': mode, 'generated_tokens': int(tail.numel()),
                       'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id),
                       'reasoning_closed': reasoning_closed if mode == 'natural' else None,
                       'final_answer': answer}
                results.append(row)
                print(json.dumps({'mode': mode, 'generated_tokens': row['generated_tokens'],
                                  'reasoning_closed': row['reasoning_closed'], 'ended': row['ended_with_eos']}),
                      flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'elapsed_seconds': time.perf_counter()-started,
              'limitations': ['Original base only; two single-turn questions.',
                              'Greedy decoding, 192-token limit; reasoning text intentionally omitted.',
                              'No memory reader, no replacement layers, no multi-turn comparison.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
