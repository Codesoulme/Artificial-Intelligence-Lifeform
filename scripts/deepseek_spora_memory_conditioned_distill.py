"""Fine-tune a SPORA layer replacement under active native memory reads."""
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

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_named_binding_probe import KEYS
from scripts.deepseek_temporal_intent_router import (CURRENT, HELD, LATER, PAST, TRAIN,
                                                       current_addresses, examples, temporal_addresses)
from scripts.deepseek_integrated_checkpoint_probe import READOUT
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.structure_transfer import SPORALayerReplacement

LAYER = 6
SOURCE_CHECKPOINT = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
START_CANDIDATE = ROOT / 'runs/deepseek-spora-block-distill-layer6-20261111/replacement.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-memory-conditioned-layer6-20261111'
STEPS = 100
SEED = 64027


def _prepare(tokenizer, row):
    inputs = closed_think_inputs(tokenizer, row['text'], 'cuda')
    if inputs['input_ids'].shape[1] > 128:
        inputs = {name: tensor[:, :128] for name, tensor in inputs.items()}
    label = row['label']
    if label == CURRENT:
        addresses, slot = current_addresses(_organism, KEYS[0], 'cuda'), 0
    else:
        event_time = 0 if label == PAST else 1
        addresses, slot = temporal_addresses(_organism, KEYS[0], event_time, 'cuda'), 1
    kwargs = {'memory_state': _state, 'memory_addresses': addresses,
              'memory_slots': torch.tensor([slot], dtype=torch.long, device='cuda'),
              'current_only': False, 'use_cache': False}
    return inputs, kwargs


def _evaluate(integrated, base, replacement, tokenizer, rows):
    original = base.model.layers[LAYER]
    output = []
    try:
        with torch.inference_mode():
            for row in rows:
                inputs, kwargs = _prepare(tokenizer, row)
                teacher = integrated(**inputs, **kwargs).logits[:, :-1].float()
                base.model.layers[LAYER] = replacement
                candidate = integrated(**inputs, **kwargs).logits[:, :-1].float()
                base.model.layers[LAYER] = original
                p = teacher.softmax(-1)
                kl = (p * (teacher.log_softmax(-1) - candidate.log_softmax(-1))).sum(-1)
                output.append({'question': row['text'], 'intent': row['label'],
                               'top1_agreement': int((teacher.argmax(-1) == candidate.argmax(-1)).sum()),
                               'scored_tokens': int(teacher.shape[1]), 'kl_sum': float(kl.sum())})
    finally:
        base.model.layers[LAYER] = original
    total = sum(row['scored_tokens'] for row in output)
    summary = {'top1_agreement': sum(row['top1_agreement'] for row in output) / total,
               'mean_token_kl_nats': sum(row['kl_sum'] for row in output) / total,
               'scored_tokens': total}
    return summary, output


def run():
    global _organism, _state
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected DeepSeek weight digest mismatch')
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
    integrated, _organism, _state = load_integrated(SOURCE_CHECKPOINT, base, MODEL, individual_device='cuda')
    integrated.reader.requires_grad_(False)
    source = torch.load(START_CANDIDATE, map_location='cpu', weights_only=True)
    if source.get('metadata', {}).get('base_weight_sha256') != WEIGHT_SHA or source['metadata'].get('layer') != LAYER:
        raise ValueError('Layer candidate provenance mismatch')
    replacement = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                        max_length=128, layer_idx=LAYER).to('cuda').train()
    replacement.load_state_dict(source['state'], strict=True)
    original_layer = base.model.layers[LAYER]
    train_rows = examples(TRAIN)
    held_rows = examples(HELD)

    # Cache teacher logits and layer-local targets under the active memory read path.
    teacher_cache = []
    with torch.inference_mode():
        for row in train_rows:
            inputs, kwargs = _prepare(tokenizer, row)
            teacher = integrated(**inputs, **kwargs, output_hidden_states=True)
            teacher_cache.append({
                'inputs': {name: value.detach() for name, value in inputs.items()},
                'kwargs': kwargs,
                'logits': teacher.logits[:, :-1].detach().to(dtype=torch.float16),
                'source_hidden': teacher.hidden_states[LAYER].detach().to(dtype=torch.float16),
                'target_hidden': teacher.hidden_states[LAYER + 1].detach().to(dtype=torch.float16),
            })

    optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-5, weight_decay=0.01, foreach=False)
    order = list(range(len(teacher_cache)))
    rng = random.Random(SEED + 1)
    rng.shuffle(order)
    trace = []
    replacement.train()
    for step in range(STEPS):
        index = order[step % len(order)]
        item = teacher_cache[index]
        teacher_logits = item['logits'].float()
        source_hidden = item['source_hidden'].float()
        target_hidden = item['target_hidden'].float()
        sequence = source_hidden.shape[1]
        positions = torch.arange(sequence, device='cuda').unsqueeze(0)
        local = replacement(source_hidden, position_ids=positions)
        relative_mse = (local - target_hidden).square().mean() / target_hidden.square().mean().clamp_min(1e-8)
        cosine_loss = 1 - F.cosine_similarity(local, target_hidden, dim=-1).mean()
        base.model.layers[LAYER] = replacement
        student_logits = integrated(**item['inputs'], **item['kwargs']).logits[:, :-1].float()
        base.model.layers[LAYER] = original_layer
        teacher_logp = (teacher_logits / 2).log_softmax(-1)
        teacher_prob = teacher_logp.exp()
        student_logp = (student_logits / 2).log_softmax(-1)
        kl = (teacher_prob * (teacher_logp - student_logp)).sum(-1).mean() * 4
        loss = 0.05 * kl + relative_mse + 0.1 * cosine_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite memory-conditioned distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(replacement.parameters(), 1.0)
        optimizer.step()
        if step in (0, 19, 39, 59, 79, 99):
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'logit_kl': float(kl.detach()),
                          'relative_mse': float(relative_mse.detach()),
                          'mean_cosine': float(F.cosine_similarity(local.detach(), target_hidden, dim=-1).mean())})

    replacement.eval()
    base.model.layers[LAYER] = original_layer
    initial = torch.load(START_CANDIDATE, map_location='cpu', weights_only=True)
    baseline_module = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                            max_length=128, layer_idx=LAYER).to('cuda').eval()
    baseline_module.load_state_dict(initial['state'], strict=True)
    before, before_rows = _evaluate(integrated, base, baseline_module, tokenizer, held_rows)
    after, after_rows = _evaluate(integrated, base, replacement, tokenizer, held_rows)
    checkpoint = OUTPUT / 'replacement.pt'
    OUTPUT.mkdir(parents=False, exist_ok=False)
    metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1',
                'base_weight_sha256': WEIGHT_SHA, 'layer': LAYER, 'width': base.config.hidden_size,
                'heads': base.config.num_attention_heads, 'max_length': 128, 'seed': SEED,
                'train_steps': STEPS, 'objective': 'native_memory_conditioned_next_token_kl'}
    atomic_save({'metadata': metadata,
                 'state': {name: tensor.detach().cpu() for name, tensor in replacement.state_dict().items()}},
                checkpoint)
    report = {
        'status': 'memory_conditioned_spora_layer6_distillation',
        'metadata': metadata,
        'base_weight_sha256': WEIGHT_SHA,
        'individual_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'start_candidate_sha256': sha(START_CANDIDATE),
        'candidate_sha256': sha(checkpoint),
        'training_examples': len(train_rows),
        'held_examples': len(held_rows),
        'held_before': before,
        'held_after': after,
        'held_rows_before': before_rows,
        'held_rows_after': after_rows,
        'loss_trace': trace,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'Twelve synthetic temporal training questions and nine authored holdout questions.',
            'One fact key, two values and one integrated individual state.',
            'The write controller and reader are frozen; only layer 6 changes.',
            'Does not establish broad memory-language behavior or multi-layer online learning.',
        ],
    }
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before': before, 'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
