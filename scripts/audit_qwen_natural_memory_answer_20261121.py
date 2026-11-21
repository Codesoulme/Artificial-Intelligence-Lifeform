"""Audit a trained Qwen natural-memory answer adapter across held values."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_addressed_full_answer import create_snapshots
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, MODEL, WEIGHT_SHA, sha
from scripts.open_base_real_memory_alignment import HELD_PAIRS, TRAIN_PAIRS
from scripts.train_qwen_natural_memory_answer_20261121 import (
    DEV_VALUES, HELD_PROMPTS, addresses_for, generate_batch,
)
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory


def run(adapter_path, output):
    adapter_path, output = adapter_path.resolve(), output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        raise ValueError('Output must be a new directory under runs')
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    snapshots, _ = create_snapshots(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='eager',
    ).to('cuda').eval()
    model.requires_grad_(False)
    payload = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if payload.get('format') != 'spora-qwen-natural-memory-answer-v1':
        raise ValueError('Unexpected adapter format')
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    adapter.load_state_dict(payload['state'], strict=True)
    adapter.eval()
    layers = [model.model.layers[5], model.model.layers[17]]
    rows = []
    for pair in HELD_PAIRS:
        for position, key in enumerate(pair):
            for template in HELD_PROMPTS:
                for value in DEV_VALUES:
                    question = template.format(key=key)
                    answers, eos = generate_batch(
                        model, tokenizer, question, snapshots[(pair, value)],
                        addresses_for(organism, key), layers, adapter)
                    expected = [value, (value + 1) % 16] if position == 0 else [
                        (value + 1) % 16, value]
                    rows.append({
                        'pair': list(pair), 'position': position, 'value_seed': value,
                        'question': question, 'expected': expected, 'answers': answers,
                        'eos': eos,
                        'correct': [answer.rstrip('。.!！ ') == str(target)
                                    for answer, target in zip(answers, expected, strict=True)],
                    })
    if (before != frozen_digest(organism, original)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL / 'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected source changed')
    output.mkdir(parents=False, exist_ok=False)
    report = {
        'status': 'natural_memory_answer_reader_holdout_audit',
        'adapter_sha256': sha(adapter_path), 'model_weight_sha256': WEIGHT_SHA,
        'individual_sha256': INDIVIDUAL_SHA, 'held_pairs': HELD_PAIRS,
        'held_prompts': HELD_PROMPTS, 'dev_values': DEV_VALUES, 'rows': rows,
        'correct_answers': sum(sum(row['correct']) for row in rows),
        'total_answers': sum(len(row['correct']) for row in rows),
        'both_correct': sum(all(row['correct']) for row in rows),
        'all_eos': sum(all(row['eos']) for row in rows),
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'The address is supplied directly by the controlled experiment.',
            'The held values are finite numeric states and do not test names, sources or unknown facts.',
            'This audit evaluates a reader adapter, not the formal runtime.',
        ],
    }
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('correct_answers', 'total_answers', 'both_correct', 'all_eos')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.adapter, args.output)
