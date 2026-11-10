"""Check cached generation with explicit SPORA memory and longer dialogue samples."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, SOURCE_CHECKPOINT
from scripts.deepseek_spora_memory_answer_distill import _load_candidate, _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced

POST = ROOT/'runs/deepseek-spora-layer18-sequence-distill-20261111/layer-18.pt'
OUTPUT = ROOT/'runs/deepseek-spora-integrated-cache-dialogue-probe-20261111.json'
GENERAL = ('如果朋友心情不好，可以怎样安慰她？', '请用简单的话解释雨是怎样形成的。')


def generate(integrated, tokenizer, prefix, memory, cache, max_tokens):
    arguments = dict(memory)
    arguments['use_cache'] = cache
    ids = integrated.generate(**prefix, **arguments, max_new_tokens=max_tokens, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tail = ids[0, prefix['input_ids'].shape[1]:]
    return {'text': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail.tolist(), 'generated_tokens': int(tail.numel()),
            'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id)}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    prior = json.loads((POST.parent/'result.json').read_text(encoding='utf-8'))
    if sha(POST) != prior['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != prior['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate provenance mismatch')
    torch.manual_seed(77318)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    key, old, new = 2, 7, 3
    for value in (old, new):
        state, decision = observe_sourced(organism, state, key, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Memory write rejected')
    rows = [
        {'intent': 'current', 'key': key, 'event_time': 1, 'value': new,
         'question': '小桐现在最喜欢什么饮料？'},
        {'intent': 'past', 'key': key, 'event_time': 0, 'value': old,
         'question': '小桐之前喜欢喝什么？'},
    ]
    original = {i: base.model.layers[i] for i in (6, 12, 18)}
    candidates = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    candidates[18] = _load_candidate(POST, 18, base).eval()
    modes = (('integrated_base', ()), ('pair', (6, 12)), ('triple', (6, 12, 18)))
    results = {}
    with torch.inference_mode():
        for mode, layers in modes:
            for i in (6, 12, 18):
                base.model.layers[i] = candidates[i] if i in layers else original[i]
            memory_results = []
            for row in rows:
                prefix = closed_think_inputs(tokenizer, row['question'], 'cuda')
                memory = _memory_args(organism, state, row)
                uncached = generate(integrated, tokenizer, prefix, memory, False, 12)
                cached = generate(integrated, tokenizer, prefix, memory, True, 12)
                memory_results.append({'intent': row['intent'], 'question': row['question'],
                                       'expected': WORDS[row['value']], 'uncached': uncached, 'cached': cached,
                                       'same_tokens': uncached['token_ids'] == cached['token_ids']})
            dialogue = []
            for question in GENERAL:
                prefix = closed_think_inputs(tokenizer, question, 'cuda')
                if prefix['input_ids'].shape[1] + 48 > 128:
                    raise ValueError('Probe would exceed replacement position limit')
                answer = generate(integrated, tokenizer, prefix, {}, True, 48)
                dialogue.append({'question': question, **answer})
            results[mode] = {'memory': memory_results, 'dialogue': dialogue}
            print(json.dumps({'mode': mode, 'memory_token_parity': [x['same_tokens'] for x in memory_results],
                              'dialogue_tokens': [x['generated_tokens'] for x in dialogue]},
                             ensure_ascii=False), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'source_sha256': sha(SOURCE_CHECKPOINT),
              'pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'layer18_sha256': sha(POST), 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Only two short controlled memory questions, same single entity.',
                              'Evaluator supplies memory addresses and temporal slot.',
                              'Two independent 48-token dialogue samples with memory disabled.',
                              'Explicitly closed reasoning prefix; no multi-turn or open-chat acceptance.',
                              'Sequence length and trained position embeddings limited to 128.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
