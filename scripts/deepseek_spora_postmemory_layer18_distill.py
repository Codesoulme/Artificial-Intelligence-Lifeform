"""Distill a third SPORA block immediately after native memory site 17."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_block_distill import _hash
from scripts.deepseek_spora_memory_answer_distill import (
    LAYERS as PAIR_LAYERS, OUTPUT as ANSWER_V3, SOURCE_CHECKPOINT,
    _facts, _load_candidate, _memory_args, _rows,
)
from scripts.deepseek_spora_memory_answer_distill_v4 import TRAIN_INDICES, VALUE_PAIRS
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.integrated_checkpoint import load_integrated
from spora.structure_transfer import SPORALayerReplacement

LAYER = 18
OUTPUT = ROOT / 'runs/deepseek-spora-postmemory-layer18-20261111'
STEPS = 320
SEED = 63109


def _evaluate(integrated, base, organism, state, candidate, original, tokenizer, rows):
    base.model.layers[LAYER] = candidate
    answers = []
    try:
        with torch.inference_mode():
            for row in rows:
                prompt = row['prefix']
                generated = integrated.generate(
                    **prompt, **_memory_args(organism, state, row), max_new_tokens=8,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(generated[0, prompt['input_ids'].shape[1]:],
                                          skip_special_tokens=True).strip()
                expected = WORDS[row['value']]
                wrong = WORDS[row['other_value']]
                answers.append({'name': row['name'], 'intent': row['intent'],
                                'question': row['question'], 'expected': expected,
                                'answer': answer, 'correct': expected in answer and wrong not in answer})
    finally:
        base.model.layers[LAYER] = original
    return {'correct': sum(row['correct'] for row in answers), 'total': len(answers),
            'accuracy': sum(row['correct'] for row in answers) / len(answers), 'rows': answers}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek base hash mismatch')
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16,
        attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    integrated, organism, _ = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    integrated.reader.requires_grad_(False)
    state, facts = _facts(organism, fresh_controlled_state(organism), VALUE_PAIRS, TRAIN_INDICES)
    train_rows, held_rows = _rows(facts, tokenizer)
    original_pair = {layer: base.model.layers[layer] for layer in PAIR_LAYERS}
    pair = {layer: _load_candidate(ANSWER_V3 / f'layer-{layer}.pt', layer, base)
            for layer in PAIR_LAYERS}
    for layer in PAIR_LAYERS:
        base.model.layers[layer] = pair[layer]
    original_layer = base.model.layers[LAYER]

    # Cache layer-18 input/output activations after both upstream replacement
    # blocks and the actual SPORA memory reads have run.
    activation_cache = []
    with torch.inference_mode():
        for row in train_rows:
            ids = torch.cat((row['prefix']['input_ids'], row['answer_ids']), dim=1)
            mask = torch.ones_like(ids)
            output = integrated(input_ids=ids, attention_mask=mask,
                                **_memory_args(organism, state, row), output_hidden_states=True)
            activation_cache.append((output.hidden_states[LAYER][0].detach().cpu().half(),
                                     output.hidden_states[LAYER + 1][0].detach().cpu().half()))

    candidate = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                      max_length=128, layer_idx=LAYER).to('cuda').train()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=7e-5, weight_decay=0.01, foreach=False)
    order = list(range(len(activation_cache)))
    rng = random.Random(SEED + 1)
    trace = []
    for step in range(STEPS):
        if step % len(order) == 0:
            rng.shuffle(order)
        source, target = activation_cache[order[step % len(order)]]
        x = source.to('cuda', dtype=torch.float32).unsqueeze(0)
        y = target.to('cuda', dtype=torch.float32).unsqueeze(0)
        position_ids = torch.arange(x.shape[1], device='cuda').unsqueeze(0)
        prediction = candidate(x, position_ids=position_ids)
        relative_mse = (prediction-y).square().mean()/y.square().mean().clamp_min(1e-8)
        cosine = F.cosine_similarity(prediction, y, dim=-1).mean()
        loss = relative_mse + 0.1*(1-cosine)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite post-memory block distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        if step in (0, 39, 79, 159, 239, 319):
            trace.append({'step': step+1, 'loss': float(loss.detach()),
                          'relative_mse': float(relative_mse.detach()), 'mean_cosine': float(cosine.detach())})

    candidate.eval()
    base.model.layers[LAYER] = original_layer
    before = _evaluate(integrated, base, organism, state, original_layer, original_layer, tokenizer, held_rows)
    after = _evaluate(integrated, base, organism, state, candidate, original_layer, tokenizer, held_rows)
    for layer in PAIR_LAYERS:
        base.model.layers[layer] = original_pair[layer]

    OUTPUT.mkdir(parents=False, exist_ok=False)
    checkpoint = OUTPUT / f'layer-{LAYER}.pt'
    metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1',
                'base_weight_sha256': WEIGHT_SHA, 'layer': LAYER, 'width': base.config.hidden_size,
                'heads': base.config.num_attention_heads, 'max_length': 128, 'seed': SEED,
                'train_steps': STEPS, 'objective': 'hidden_distillation_after_native_memory_and_spora_pair'}
    torch.save({'metadata': metadata,
                'state': {name: tensor.detach().cpu() for name, tensor in candidate.state_dict().items()}},
               checkpoint)
    report = {
        'status': 'post_memory_site_layer18_spora_distillation',
        'metadata': metadata,
        'base_weight_sha256': WEIGHT_SHA,
        'source_integrated_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'upstream_pair_sha256': {str(layer): sha(ANSWER_V3 / f'layer-{layer}.pt') for layer in PAIR_LAYERS},
        'train_entities': [fact['name'] for fact in facts if fact['train_split']],
        'held_entities': [fact['name'] for fact in facts if not fact['train_split']],
        'training_sequences': len(activation_cache),
        'held_sequences': len(held_rows),
        'held_base_pair': before,
        'held_with_layer18': after,
        'loss_trace': trace,
        'candidate_sha256': sha(checkpoint),
        'elapsed_seconds': time.perf_counter()-started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Layer 18 is trained to match local activations under the paired blocks and active memory, not directly on answer loss.',
            'Six training entities and two held entities, one finite drink-memory domain.',
            'This is a third-layer post-memory path probe, not a full backbone conversion.',
        ],
    }
    with (OUTPUT/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'held_before': before['accuracy'], 'held_after': after['accuracy']},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
