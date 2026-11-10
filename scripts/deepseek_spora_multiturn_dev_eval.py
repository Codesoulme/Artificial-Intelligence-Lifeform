"""Run held development conversations without a SPORA memory state."""
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
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_integrated_cache_dialogue_probe import POST

DATA = ROOT/'data/conversation-development-v2.json'
OUTPUT = ROOT/'runs/deepseek-spora-multiturn-dev-eval-20261111.json'
POSITION_LIMIT = 256
MAX_NEW = 48


def make_prefix(tokenizer, messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.rstrip().endswith('<think>'):
        raise ValueError('Unexpected DeepSeek assistant prefix')
    return tokenizer(prompt + '</think>\n\n', return_tensors='pt', add_special_tokens=False).to('cuda')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base weight hash mismatch')
    prior = json.loads((POST.parent/'result.json').read_text(encoding='utf-8'))
    if sha(POST) != prior['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != prior['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate provenance mismatch')
    data = json.loads(DATA.read_text(encoding='utf-8'))
    if len(data['scenarios']) != 8 or any(len(s['turns']) != 3 for s in data['scenarios']):
        raise ValueError('Unexpected development set')
    torch.manual_seed(58331)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    original = {i: base.model.layers[i] for i in (6, 12, 18)}
    candidates = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    candidates[18] = _load_candidate(POST, 18, base).eval()
    for candidate in candidates.values():
        candidate.extend_positions(POSITION_LIMIT)
    results = {}
    with torch.inference_mode():
        for mode, layers in (('original', ()), ('extended_pair', (6, 12)),
                             ('extended_triple', (6, 12, 18))):
            for i in (6, 12, 18):
                base.model.layers[i] = candidates[i] if i in layers else original[i]
            scenarios = []
            for scenario in data['scenarios']:
                messages, turns = [], []
                for question in scenario['turns']:
                    messages.append({'role': 'user', 'content': question})
                    prefix = make_prefix(tokenizer, messages)
                    prefix_len = int(prefix['input_ids'].shape[1])
                    budget = min(MAX_NEW, POSITION_LIMIT-prefix_len)
                    if budget < 1:
                        turns.append({'question': question, 'prefix_tokens': prefix_len,
                                      'skipped': 'position_limit'})
                        break
                    generated = base.generate(**prefix, max_new_tokens=budget, use_cache=True,
                                              do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                              eos_token_id=tokenizer.eos_token_id)
                    tail = generated[0, prefix_len:]
                    answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
                    ended = bool(tail.numel() and tail[-1] == tokenizer.eos_token_id)
                    turns.append({'question': question, 'answer': answer, 'prefix_tokens': prefix_len,
                                  'generated_tokens': int(tail.numel()), 'max_new_tokens': budget,
                                  'ended_with_eos': ended, 'truncated': not ended and tail.numel() == budget})
                    messages.append({'role': 'assistant', 'content': answer})
                scenarios.append({'name': scenario['name'], 'turns': turns})
                print(json.dumps({'mode': mode, 'scenario': scenario['name'], 'turns': len(turns),
                                  'truncated': sum(t.get('truncated', False) for t in turns)},
                                 ensure_ascii=False), flush=True)
            results[mode] = scenarios
    report = {'results': results, 'data_sha256': sha(DATA), 'base_sha256': WEIGHT_SHA,
              'pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)}, 'layer18_sha256': sha(POST),
              'position_extension': 'zero after 127, no position-tail training',
              'elapsed_seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Previously existing development set; not independent final acceptance.',
                              'Each model sees its own earlier generated answers; conversation histories differ.',
                              'Memory reader is disabled; this measures context-based dialogue only.',
                              'Closed reasoning prefix; greedy decoding; max 48 new tokens per turn.',
                              'New SPORA positions from 128 onward are untrained zeros.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
