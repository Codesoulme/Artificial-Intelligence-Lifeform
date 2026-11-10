"""Evaluate the natural-sequence-distilled pair under sampled reasoning."""
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
from scripts.deepseek_official_sampling_probe import QUESTIONS
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate

NEW = ROOT/'runs/deepseek-spora-natural-hidden-distill-20261111'
OUTPUT = ROOT/'runs/deepseek-spora-natural-distill-generation-eval-20261111.json'
SEED = 4830
MAX_NEW = 512


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    training = json.loads((NEW/'result.json').read_text(encoding='utf-8'))
    for layer in (6, 12):
        info = training['layers'][str(layer)]
        root_hash = info.get('root_sha256', info['start_sha256'])
        if sha(NEW/f'layer-{layer}.pt') != info['candidate_sha256'] or sha(PAIR/f'layer-{layer}.pt') != root_hash:
            raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    for i in (6, 12):
        base.model.layers[i] = _load_candidate(NEW/f'layer-{i}.pt', i, base).eval()
    results = []
    with torch.inference_mode():
        for question in QUESTIONS:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                                   add_generation_prompt=True)
            if not prompt.rstrip().endswith('<think>'):
                raise ValueError('Unexpected chat template')
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            if inputs['input_ids'].shape[1]+MAX_NEW > 768:
                raise ValueError('Generation exceeds trained position table')
            torch.manual_seed(SEED)
            torch.cuda.synchronize()
            before = time.perf_counter()
            ids = base.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.95,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tail = ids[0, inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(tail, skip_special_tokens=True)
            closed = '</think>' in text
            row = {'question': question, 'seed': SEED, 'generated_tokens': int(tail.numel()),
                   'reasoning_closed': closed,
                   'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id),
                   'final_answer': text.split('</think>', 1)[1].strip() if closed else '',
                   'elapsed_seconds': time.perf_counter()-before}
            results.append(row)
            print(json.dumps({k: row[k] for k in ('seed', 'generated_tokens', 'reasoning_closed',
                                                   'ended_with_eos', 'elapsed_seconds')}), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA,
              'pair_sha256': {str(i): sha(NEW/f'layer-{i}.pt') for i in (6, 12)},
              'teacher_data_sha256': training.get('teacher_data_sha256', training.get('natural_data_sha256')),
              'temperature': 0.6, 'top_p': 0.95,
              'max_new_tokens': MAX_NEW, 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two previously inspected questions with one seed each.',
                              'No SPORA memory reader; reasoning text intentionally omitted.',
                              'Teacher samples and maximum generation were capped at 512 tokens.',
                              'Natural long-position training used only two incomplete trajectories.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
