"""Audit exact numeric query routing and unknown-address rejection."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_addressed_full_answer import create_snapshots
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, MODEL, WEIGHT_SHA, sha
from scripts.open_base_real_memory_alignment import HELD_PAIRS, TRAIN_PAIRS
from scripts.train_qwen_natural_memory_answer_20261121 import (
    HELD_PROMPTS, addresses_for, generate_batch,
)
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory
from spora.open_base_query_route import RecordedAddressIndex, resolve_numeric_object


ANSWER_RUN = ROOT / 'runs/qwen-natural-memory-answer-v1-20261121'


def run(output):
    output = output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        raise ValueError('Output must be a new directory under runs')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    snapshots, _ = create_snapshots(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    index = RecordedAddressIndex.from_keys(organism.config.key_count, [*HELD_PAIRS[0], *HELD_PAIRS[1]])
    known_rows = []
    unknown_rows = []
    for pair in HELD_PAIRS:
        for position, key in enumerate(pair):
            for template in HELD_PROMPTS:
                question = template.format(key=key)
                known_rows.append({'question': question, 'expected_key': key,
                                   'resolved_key': resolve_numeric_object(question, index),
                                   'accepted': resolve_numeric_object(question, index) is not None})
    for key in (12, 13, 14):
        question = HELD_PROMPTS[0].format(key=key)
        unknown_rows.append({'question': question, 'expected_key': key,
                             'resolved_key': resolve_numeric_object(question, index),
                             'accepted': resolve_numeric_object(question, index) is not None})

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='eager',
    ).to('cuda').eval()
    model.requires_grad_(False)
    payload = torch.load(ANSWER_RUN / 'adapter.pt', map_location='cpu', weights_only=True)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    adapter.load_state_dict(payload['state'], strict=True)
    adapter.eval()
    layers = [model.model.layers[5], model.model.layers[17]]
    answer_rows = []
    for pair in HELD_PAIRS:
        for position, key in enumerate(pair):
            for template in HELD_PROMPTS:
                question = template.format(key=key)
                routed = resolve_numeric_object(question, index)
                generated, eos = generate_batch(
                    model, tokenizer, question, snapshots[(pair, 1)],
                    addresses_for(organism, routed), layers, adapter)
                expected = [1, 2] if position == 0 else [2, 1]
                answer_rows.append({
                    'question': question, 'expected_key': key, 'routed_key': routed,
                    'answers': generated, 'eos': eos,
                    'correct': [answer.rstrip('。.!！ ') == str(target)
                                for answer, target in zip(generated, expected, strict=True)],
                })
    if (before != frozen_digest(organism, original)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL / 'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected source changed')
    output.mkdir(parents=False, exist_ok=False)
    report = {
        'status': 'qwen_numeric_presence_guard_candidate',
        'index': index.payload(), 'known_rows': known_rows,
        'unknown_rows': unknown_rows, 'answer_rows': answer_rows,
        'known_accepted': sum(row['accepted'] for row in known_rows),
        'known_total': len(known_rows),
        'unknown_rejected': sum(not row['accepted'] for row in unknown_rows),
        'unknown_total': len(unknown_rows),
        'answer_correct': sum(sum(row['correct']) for row in answer_rows),
        'answer_total': sum(len(row['correct']) for row in answer_rows),
        'all_eos': sum(all(row['eos']) for row in answer_rows),
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'The index is an exact routing registry and not a learned identity model.',
            'The numeric parser is limited to one explicit object or number mention.',
            'The answer adapter still uses finite controlled numeric states.',
            'The candidate is not integrated into a durable formal checkpoint.',
        ],
    }
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'known_accepted', 'known_total', 'unknown_rejected', 'unknown_total',
        'answer_correct', 'answer_total', 'all_eos')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
