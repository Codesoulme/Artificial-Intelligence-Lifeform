"""Compare cached and full-prefix generation with one frozen memory snapshot."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

OUTPUT = ROOT/'runs/open-base-cache-probe-20261109.json'


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    torch.manual_seed(23701)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    organism, state, _ = load(INDIVIDUAL, 'cuda')
    before = frozen_digest(organism, state)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    layers = [model.model.layers[5], model.model.layers[17]]
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': '你好，请简短介绍你的能力。'}],
                                               tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors='pt').to('cuda')
    def generate(cache):
        return model.generate(**inputs, max_new_tokens=24, do_sample=False, use_cache=cache,
                              pad_token_id=tokenizer.eos_token_id)[0].cpu().tolist()
    with torch.inference_mode():
        baseline = generate(True)
        with memory_hooks(layers, adapter, state):
            zero = generate(True)
        if baseline != zero:
            raise AssertionError('Zero adapter changed greedy generation')
        for out in adapter.out:
            torch.nn.init.normal_(out.weight, std=0.02)
        with memory_hooks(layers, adapter, state):
            cached = generate(True)
            full = generate(False)
        if generate(True) != baseline:
            raise AssertionError('Removing hooks changed generation')
    if sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(MODEL/'model.safetensors') != WEIGHT_SHA or frozen_digest(organism, state) != before:
        raise ValueError('Protected input changed')
    report = {'status': 'cached_generation_structural_check', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'layers': [5, 17],
              'zero_adapter_matches_baseline': baseline == zero, 'cached_matches_full_prefix': cached == full,
              'baseline_new_tokens': len(baseline)-inputs.input_ids.shape[1],
              'cached_new_tokens': len(cached)-inputs.input_ids.shape[1],
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter()-start,
              'limitations': ['One prompt and one fixed memory snapshot; this is not a general cache proof.',
                              'Nonzero adapter is randomly perturbed and untrained.',
                              'Switching memory requires a fresh generation context and fresh KV cache.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
