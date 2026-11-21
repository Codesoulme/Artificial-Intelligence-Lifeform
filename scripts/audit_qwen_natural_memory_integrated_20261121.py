"""End-to-end audit of Qwen natural query routing plus SPORA memory readout."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

from scripts import train_qwen_memory_intent_gate_20261121 as gate_module
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, MODEL, WEIGHT_SHA, sha
from scripts.open_base_real_memory_alignment import HELD_PAIRS, TRAIN_PAIRS, variants
from scripts.open_base_addressed_full_answer import create_snapshots
from scripts.train_qwen_natural_memory_address_20261121 import (
    TokenAddressHead, contextual_content_features, pad_token_features, pad_token_masks,
)
from scripts.train_qwen_natural_memory_answer_20261121 import (
    HELD_PROMPTS, addresses_for, generate_batch,
)
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory


CONTEXT_RUN = ROOT / 'runs/qwen-natural-memory-context-address-v1-20261121'
ANSWER_RUN = ROOT / 'runs/qwen-natural-memory-answer-v1-20261121'
GATE_RUN = ROOT / 'runs/qwen-memory-intent-gate-v9-20261121'


def run(output):
    output = output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output.exists():
        raise ValueError('Output must be a new directory under runs')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='eager',
    ).to('cuda').eval()
    model.requires_grad_(False)
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    states, _ = variants(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    snapshots, _ = create_snapshots(organism, original, TRAIN_PAIRS + HELD_PAIRS)

    context_payload = torch.load(CONTEXT_RUN / 'context_address_head.pt', map_location='cpu', weights_only=True)
    answer_payload = torch.load(ANSWER_RUN / 'adapter.pt', map_location='cpu', weights_only=True)
    gate_payload = torch.load(GATE_RUN / 'intent_gate.pt', map_location='cpu', weights_only=True)
    if answer_payload.get('format') != 'spora-qwen-natural-memory-answer-v1':
        raise ValueError('Unexpected answer adapter format')
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    adapter.load_state_dict(answer_payload['state'], strict=True)
    adapter.eval()
    context_head = TokenAddressHead(model.config.hidden_size).cuda()
    context_head.load_state_dict(context_payload['state'], strict=True)
    context_head.eval().requires_grad_(False)
    intent_gate = gate_module.IntentGate(model.config.hidden_size * 2, gate_payload['cutoff']).cuda()
    intent_gate.load_state_dict(gate_payload['state'], strict=True)
    intent_gate.eval().requires_grad_(False)
    runtime_gate = gate_module.make_runtime_gate(intent_gate, gate_payload['cutoff'])
    layers = [model.model.layers[5], model.model.layers[17]]

    questions = [(template.format(key=key), pair, position, key)
                 for pair in HELD_PAIRS for position, key in enumerate(pair)
                 for template in HELD_PROMPTS]
    context_rows, context_attention_rows = contextual_content_features(
        model, tokenizer, [question for question, _, _, _ in questions])
    context_features, context_mask = pad_token_features(context_rows)
    context_attention = pad_token_masks(context_attention_rows)
    gate_features = gate_module.hidden_features(
        model, tokenizer, [question for question, _, _, _ in questions])
    with torch.inference_mode():
        tens, ones, _ = context_head(context_features, context_mask, context_attention)
        predicted = (tens.argmax(-1) * 10 + ones.argmax(-1)).cpu().tolist()
        gate_scores = intent_gate(gate_features).sigmoid().cpu().tolist()

    rows = []
    for (question, pair, position, expected_key), predicted_key, gate_score in zip(
            questions, predicted, gate_scores, strict=True):
        generated, eos = generate_batch(
            model, tokenizer, question, snapshots[(pair, 1)],
            addresses_for(organism, predicted_key), layers, adapter)
        expected = [1, 2] if position == 0 else [2, 1]
        rows.append({
            'question': question, 'pair': list(pair), 'expected_key': expected_key,
            'predicted_key': predicted_key, 'address_correct': predicted_key == expected_key,
            'memory_score': gate_score, 'memory_enabled': gate_score >= gate_payload['cutoff'],
            'expected': expected, 'answers': generated, 'eos': eos,
            'correct': [answer.rstrip('。.!！ ') == str(target)
                        for answer, target in zip(generated, expected, strict=True)],
        })

    if (before != frozen_digest(organism, original)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(MODEL / 'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected source changed')
    output.mkdir(parents=False, exist_ok=False)
    report = {
        'status': 'qwen_spora_natural_memory_end_to_end_candidate',
        'context_run': str(CONTEXT_RUN), 'answer_run': str(ANSWER_RUN),
        'gate_run': str(GATE_RUN), 'rows': rows,
        'address_correct': sum(row['address_correct'] for row in rows),
        'address_total': len(rows),
        'memory_enabled': sum(row['memory_enabled'] for row in rows),
        'answer_correct': sum(sum(row['correct']) for row in rows),
        'answer_total': sum(len(row['correct']) for row in rows),
        'both_answers_correct': sum(all(row['correct']) for row in rows),
        'all_eos': sum(all(row['eos']) for row in rows),
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'The test uses finite numeric memory states and direct controlled entity addresses in the answer adapter.',
            'The address head and answer adapter were trained on related synthetic numeric tasks.',
            'Ordinary non-memory prompts, unknown facts, writing, correction and reload are not tested here.',
            'This candidate is not installed into the formal runtime or individual.',
        ],
    }
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'address_correct', 'address_total', 'memory_enabled', 'answer_correct',
        'answer_total', 'both_answers_correct', 'all_eos')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
