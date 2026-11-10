"""Compare SPORA replacement blocks with a bounded natural-reasoning sample."""
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
from scripts.deepseek_spora_integrated_cache_dialogue_probe import POST

OUTPUT = ROOT/'runs/deepseek-spora-natural-sampling-probe-20261111.json'
SEED = 4830
MAX_NEW = 512
POSITION_LIMIT = 768


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight hash mismatch')
    previous = json.loads((POST.parent/'result.json').read_text(encoding='utf-8'))
    if sha(POST) != previous['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != previous['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    originals = {i: base.model.layers[i] for i in (6, 12, 18)}
    candidates = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    candidates[18] = _load_candidate(POST, 18, base).eval()
    for candidate in candidates.values():
        candidate.extend_positions(POSITION_LIMIT)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': QUESTIONS[0]}], tokenize=False,
                                           add_generation_prompt=True)
    if not prompt.rstrip().endswith('<think>'):
        raise ValueError('Unexpected chat template')
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    if inputs['input_ids'].shape[1]+MAX_NEW > POSITION_LIMIT:
        raise ValueError('Prompt and generation exceed position limit')
    results = {}
    with torch.inference_mode():
        for mode, replacements in (('pair', (6, 12)), ('triple', (6, 12, 18))):
            for i in (6, 12, 18):
                base.model.layers[i] = candidates[i] if i in replacements else originals[i]
            torch.manual_seed(SEED)
            torch.cuda.synchronize()
            before = time.perf_counter()
            generated = base.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                                      temperature=0.6, top_p=0.95, use_cache=True,
                                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tail = generated[0, inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(tail, skip_special_tokens=True)
            closed = '</think>' in text
            results[mode] = {'generated_tokens': int(tail.numel()), 'reasoning_closed': closed,
                             'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id),
                             'final_answer': text.split('</think>', 1)[1].strip() if closed else '',
                             'elapsed_seconds': time.perf_counter()-before}
            print(json.dumps({'mode': mode, **{k: v for k, v in results[mode].items() if k != 'final_answer'}}),
                  flush=True)
    report = {'question': QUESTIONS[0], 'seed': SEED, 'results': results,
              'base_sha256': WEIGHT_SHA, 'pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'layer18_sha256': sha(POST), 'temperature': 0.6, 'top_p': 0.95, 'max_new_tokens': MAX_NEW,
              'position_extension': 'zero beyond 127 up to 767; no long-position training',
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only one previously inspected question and one random seed per candidate.',
                              'New position vectors are untrained; this is not a long-context model.',
                              'No SPORA memory reader; reasoning text intentionally omitted.',
                              'A 512-token cap is below the official evaluation budget.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
