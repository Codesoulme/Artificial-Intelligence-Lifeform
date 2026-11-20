"""Check ordinary Qwen chat with the finite memory reader disabled and enabled."""

import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

from audit_awake_reasoning_residual_holdout_20261121 import load_scenarios
from deepseek_output_aligned_memory import ROOT, sha, torch, write
from scripts.open_base_addressed_full_answer import create_snapshots
from scripts.open_base_baseline import INDIVIDUAL, MODEL, INDIVIDUAL_SHA, WEIGHT_SHA
from spora.growing import load
from spora.open_base_memory import OpenBaseMemory, memory_hooks


ADAPTER = ROOT / 'runs/open-base-addressed-full-answer-20261109/adapter.pt'
PROMPTS = tuple(scenario['turns'][0] for scenario in load_scenarios())


@torch.inference_mode()
def generate(model, tokenizer, question, *, reader=None, state=None, layers=None, limit=64):
    formatted = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors='pt', add_special_tokens=False).to('cuda')
    if reader is None:
        output = model.generate(**inputs, do_sample=False, max_new_tokens=limit, use_cache=True,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    else:
        with memory_hooks(layers, reader, state, current_only=True):
            output = model.generate(**inputs, do_sample=False, max_new_tokens=limit, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tail = output[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(tail, skip_special_tokens=True).strip(), int(tail.numel()), bool(
        tail.numel() and tail[-1] == tokenizer.eos_token_id)


def main(output):
    if output.exists():
        raise FileExistsError(output)
    paths = (MODEL / 'model.safetensors', INDIVIDUAL, ADAPTER,
             MODEL / 'config.json', MODEL / 'tokenizer.json', MODEL / 'tokenizer_config.json')
    protected = {path: sha(path) for path in paths}
    started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    organism, original, _ = load(INDIVIDUAL, 'cpu')
    snapshots, _ = create_snapshots(organism, original, ((8, 9),))
    state = snapshots[((8, 9), 1)]
    state = SimpleNamespace(current=[matrix[:1].contiguous() for matrix in state.current], pages=[])
    payload = torch.load(ADAPTER, map_location='cpu', weights_only=True)
    if payload.get('format') != 'spora-open-base-p26-full-answer-v1':
        raise ValueError('Unexpected memory adapter format')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    reader = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    reader.load_state_dict(payload['state'], strict=True)
    reader.eval()
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    for question in PROMPTS:
        plain = generate(model, tokenizer, question)
        active = generate(model, tokenizer, question, reader=reader, state=state, layers=layers)
        rows.append({'question': question, 'memory_off': plain, 'memory_on': active,
                     'tokens_changed': plain[0] != active[0]})
        print(question, {'off': plain[0], 'on': active[0]}, flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected source changed during the audit')
    write(output, {
        'format': 'qwen-memory-chat-interference-v1',
        'model': str(MODEL),
        'adapter': str(ADAPTER),
        'rows': rows,
        'sources': {str(path): digest for path, digest in protected.items()},
        'individual_sha256': INDIVIDUAL_SHA,
        'weight_sha256': WEIGHT_SHA,
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'The reader was trained for finite numbered memory questions, not ordinary language.',
            'Memory was manually activated with one controlled state; no natural router was evaluated.',
            'Different output under activation is interference evidence, not semantic quality evidence.',
        ],
    })
    print('Interference report written', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
