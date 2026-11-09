"""Test real SPORA matrix reads inside a local pretrained Qwen; no training."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, REVISION, WEIGHT_SHA, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks
from spora.sleep_memory import history_site

OUTPUT = ROOT/'runs/open-base-memory-probe-20261109.json'


def zero_state(state):
    current = [torch.zeros_like(matrix) for matrix in state.current]
    pages = [[[torch.zeros_like(t) for t in history_site(page, site)] for site in range(len(current))]
             for page in state.pages]
    return SimpleNamespace(current=current, pages=pages)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Model or individual digest mismatch')
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
    prompt = '对象12现在的数值是多少？'
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        baseline = model(**inputs, use_cache=False).logits.float()
        with memory_hooks(layers, adapter, state):
            zero_output = model(**inputs, use_cache=False).logits.float()
        if not torch.equal(baseline, zero_output):
            raise AssertionError('Zero-initialized adapter changed baseline')
        # Deliberate test perturbation, not a trained or usable memory adapter.
        for out in adapter.out:
            torch.nn.init.normal_(out.weight, std=0.02)
        with memory_hooks(layers, adapter, state):
            real_output = model(**inputs, use_cache=False).logits.float()
        with memory_hooks(layers, adapter, zero_state(state)):
            empty_output = model(**inputs, use_cache=False).logits.float()
        with_memory_change = (real_output-empty_output).abs()
        if not bool(torch.isfinite(real_output).all()) or float(with_memory_change.max()) == 0:
            raise AssertionError('Memory perturbation was nonfinite or had no effect')
        unhooked = model(**inputs, use_cache=False).logits.float()
        if not torch.equal(baseline, unhooked):
            raise AssertionError('Hook removal did not restore baseline')
    if (before != frozen_digest(organism, state) or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL/'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected input changed')
    result = {'status': 'structural_probe_not_semantic_memory', 'model': 'Qwen/Qwen2.5-0.5B-Instruct',
              'revision': REVISION, 'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'individual_frozen_digest': before, 'layers': [5, 17], 'seed': 23701,
              'zero_adapter_logits_exact': True, 'hook_removal_logits_exact': True,
              'real_vs_zero_memory_max_logit_change': float(with_memory_change.max()),
              'real_vs_zero_memory_mean_logit_change': float(with_memory_change.mean()),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter()-start,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'limitations': ['The adapter is untrained and deliberately perturbed; the logit change is not an answer.',
                              'No semantics, knowledge preservation or long-term memory behavior is established.',
                              'State is a read-only snapshot, not a working online write mechanism.',
                              'No pretrained weight or original individual is overwritten.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
