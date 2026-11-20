"""Measure the locally pinned Qwen2.5-0.5B-Instruct chat baseline."""

import argparse
import json
import time
from pathlib import Path

from audit_awake_reasoning_residual_holdout_20261121 import load_scenarios
from deepseek_output_aligned_memory import ROOT, sha, torch, write


MODEL_DIR = ROOT / 'runs/open-base-qwen2.5-0.5b-7ae5576'
DATA = ROOT / 'data/conversation-development-v2.json'


@torch.inference_mode()
def run_dialogue(model, tokenizer, scenario, limit, seed):
    history = []
    rows = []
    for turn_index, question in enumerate(scenario['turns']):
        history.append({'role': 'user', 'content': question})
        text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        torch.manual_seed(seed + turn_index)
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=limit,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        tail = generated[0, inputs['input_ids'].shape[1]:]
        answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
        rows.append({
            'turn': turn_index + 1,
            'question': question,
            'answer': answer,
            'prompt_tokens': int(inputs['input_ids'].shape[1]),
            'generated_tokens': int(tail.numel()),
            'eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id),
        })
        history.append({'role': 'assistant', 'content': answer})
    return rows


def main(output, limit):
    if output.exists():
        raise FileExistsError(output)
    if not MODEL_DIR.is_dir() or not DATA.is_file():
        raise FileNotFoundError('Pinned Qwen model or holdout data is missing')
    protected = {MODEL_DIR / name: sha(MODEL_DIR / name)
                 for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json')}
    protected[DATA] = sha(DATA)
    started = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='sdpa',
    ).to('cuda').eval()
    model.requires_grad_(False)
    rows = []
    for scenario_index, scenario in enumerate(load_scenarios()):
        dialogue = run_dialogue(model, tokenizer, scenario, limit, 34127 + scenario_index * 100)
        rows.append({'variant': 'qwen2.5-0.5b-instruct', 'scenario': scenario['name'], 'turns': dialogue})
        print(scenario['name'], [row['answer'] for row in dialogue], flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('A protected model or holdout file changed during the audit')
    write(output, {
        'format': 'qwen2.5-0.5b-instruct-holdout-v1',
        'model_dir': str(MODEL_DIR),
        'data': str(DATA),
        'max_new_tokens': limit,
        'rows': rows,
        'sources': {str(path): digest for path, digest in protected.items()},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Development holdout only; semantic quality still requires manual review.',
            'Greedy decoding is used for reproducibility.',
            'No SPORA memory is connected in this baseline.',
        ],
    })
    print('Qwen holdout report written', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('max-new-tokens must be positive')
    main(args.output, args.max_new_tokens)
