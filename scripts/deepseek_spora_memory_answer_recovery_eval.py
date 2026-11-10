"""Recover held-answer evaluation for saved answer-distilled SPORA blocks."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import (
    LAYERS, NAMES, OUTPUT, SOURCE_CHECKPOINT, START_CANDIDATES,
    _evaluate, _facts, _load_candidate, _rows,
)
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated

RESULT = OUTPUT / 'recovered-evaluation.json'


def run():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    state, facts = _facts(organism, fresh_controlled_state(organism))
    _, held_rows = _rows(facts, tokenizer)
    originals = {layer: base.model.layers[layer] for layer in LAYERS}
    initial = {layer: _load_candidate(path, layer, base) for layer, path in START_CANDIDATES.items()}
    trained_paths = {layer: OUTPUT / f'layer-{layer}.pt' for layer in LAYERS}
    trained = {layer: _load_candidate(path, layer, base) for layer, path in trained_paths.items()}
    baseline = _evaluate(integrated, base, organism, state, originals, originals, tokenizer, held_rows)
    before = _evaluate(integrated, base, organism, state, initial, originals, tokenizer, held_rows)
    after = _evaluate(integrated, base, organism, state, trained, originals, tokenizer, held_rows)
    report = {
        'status': 'recovered_saved_answer_distillation_candidate_evaluation',
        'base_weight_sha256': WEIGHT_SHA,
        'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'held_entities': list(NAMES[6:8]),
        'held_answers': len(held_rows),
        'baseline': baseline,
        'paired_candidate_before_answer_distillation': before,
        'saved_candidate_after_answer_distillation': after,
        'candidate_sha256': {str(layer): sha(path) for layer, path in trained_paths.items()},
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'The training process saved candidate checkpoints but failed while writing its result due to a missing report-only import.',
            'The exact per-step training trace and training-time teacher generations were not persisted and cannot be recovered.',
            'This evaluation reconstructs the same fixed eight-entity fact schedule and scores the two held entities.',
            'Twelve short authored prompts and exact expected-word scoring are not open-chat acceptance.',
        ],
    }
    with RESULT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'baseline': baseline['accuracy'], 'before': before['accuracy'],
                      'after': after['accuracy']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
