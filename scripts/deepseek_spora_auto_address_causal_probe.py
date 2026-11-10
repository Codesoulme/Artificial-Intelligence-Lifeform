"""Check token-level native memory queries in cached and uncached generation."""
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
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, SOURCE_CHECKPOINT, _load_candidate
from scripts.deepseek_spora_integrated_cache_dialogue_probe import POST, generate
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.internal_memory import observe_sourced

OUTPUT = ROOT/'runs/deepseek-spora-auto-address-causal-probe-20261111.json'
QUESTIONS = ('你能告诉我一件以前记录的事情吗？', '请简短说明你现在想做什么。')


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
    torch.set_num_threads(2)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state = fresh_controlled_state(organism)
    for value in (7, 3):
        state, decision = observe_sourced(organism, state, 2, value, 'direct', 1.0)
        if not decision['current_write']:
            raise ValueError('Memory write rejected')
    originals = {i: base.model.layers[i] for i in (6, 12, 18)}
    replacements = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    replacements[18] = _load_candidate(POST, 18, base).eval()
    results = {}
    with torch.inference_mode():
        for mode, layers in (('integrated_base', ()), ('pair', (6, 12)), ('triple', (6, 12, 18))):
            for i in (6, 12, 18):
                base.model.layers[i] = replacements[i] if i in layers else originals[i]
            rows = []
            for question in QUESTIONS:
                prefix = closed_think_inputs(tokenizer, question, 'cuda')
                memory = {'memory_state': state, 'current_only': False}
                uncached = generate(integrated, tokenizer, prefix, memory, False, 16)
                cached = generate(integrated, tokenizer, prefix, memory, True, 16)
                rows.append({'question': question, 'uncached': uncached, 'cached': cached,
                             'same_tokens': uncached['token_ids'] == cached['token_ids']})
            results[mode] = rows
            print(json.dumps({'mode': mode, 'token_parity': [x['same_tokens'] for x in rows]}), flush=True)
    report = {'results': results, 'base_sha256': WEIGHT_SHA, 'source_sha256': sha(SOURCE_CHECKPOINT),
              'pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'layer18_sha256': sha(POST), 'elapsed_seconds': time.perf_counter()-started,
              'limitations': ['Two short prompts and 16 generated tokens, no correctness scoring.',
                              'Automatic token query is not a verified language-to-memory router.',
                              'One synthetic state with current and historical memory pages.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
