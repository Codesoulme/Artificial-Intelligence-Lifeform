"""Bounded original-base probe using model-card sampling recommendations."""
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

OUTPUT = ROOT/'runs/deepseek-official-sampling-probe-20261111.json'
QUESTIONS = ('我上午先去了邮局，下午又去了菜市场。上午去了哪里？',
             '小叶不喝咖啡，她喜欢喝温水。她喜欢什么？')
SEEDS = (4830, 4831)
MAX_NEW = 512


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    results = []
    with torch.inference_mode():
        for question in QUESTIONS:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                                   add_generation_prompt=True)
            if not prompt.rstrip().endswith('<think>'):
                raise ValueError('Unexpected chat template')
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            for seed in SEEDS:
                torch.manual_seed(seed)
                torch.cuda.synchronize()
                before = time.perf_counter()
                generated = base.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                                          temperature=0.6, top_p=0.95, use_cache=True,
                                          pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-before
                tail = generated[0, inputs['input_ids'].shape[1]:]
                text = tokenizer.decode(tail, skip_special_tokens=True)
                closed = '</think>' in text
                answer = text.split('</think>', 1)[1].strip() if closed else ''
                row = {'question': question, 'seed': seed, 'generated_tokens': int(tail.numel()),
                       'reasoning_closed': closed,
                       'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id),
                       'final_answer': answer, 'elapsed_seconds': elapsed}
                results.append(row)
                print(json.dumps({k: row[k] for k in ('seed', 'generated_tokens', 'reasoning_closed',
                                                       'ended_with_eos', 'elapsed_seconds')}), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA,
              'settings': {'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': MAX_NEW,
                           'seeds': SEEDS, 'system_prompt': False, 'reasoning_prefix': 'chat template'},
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two previously inspected development questions and two seeds, not a benchmark.',
                              '512-token cap is below the model-card evaluation budget.',
                              'No SPORA layers or memory reader; reasoning text intentionally omitted.',
                              'FP16 CUDA results may differ from the original BF16 weights.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
