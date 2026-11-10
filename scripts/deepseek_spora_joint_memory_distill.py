"""Jointly distill two SPORA blocks while native SPORA memory remains active."""
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
from spora.integrated_checkpoint import load_integrated
from spora.language_training import atomic_save
from spora.structure_transfer import SPORALayerReplacement

SOURCE_CHECKPOINT = ROOT / 'runs/deepseek-temporal-intent-router-native-spora-site-20261110/historical-individual.pt'
START_CANDIDATES = {
    6: ROOT / 'runs/deepseek-spora-memory-conditioned-layer6-20261111/replacement.pt',
    12: ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt',
}
OUTPUT = ROOT / 'runs/deepseek-spora-joint-memory-distill-v3-20261111'
START_DIR = ROOT / 'runs/deepseek-spora-joint-memory-distill-v2-20261111'
LAYERS = (6, 12)
STEPS = 120
SEED = 28741


def _inputs(tokenizer, row):
    inputs = closed_think_inputs(tokenizer, row['text'], 'cuda')
    if inputs['input_ids'].shape[1] > 128:
        inputs = {name: value[:, :128] for name, value in inputs.items()}
    if row['label'] == CURRENT:
        addresses = current_addresses(_organism, KEYS[0], 'cuda')
        slot = 0
    else:
        event_time = 0 if row['label'] == PAST else 1
        addresses = temporal_addresses(_organism, KEYS[0], event_time, 'cuda')
        slot = 1
    controls = {'memory_state': _state, 'memory_addresses': addresses,
                'memory_slots': torch.tensor([slot], dtype=torch.long, device='cuda'),
                'current_only': False, 'use_cache': False}
    return inputs, controls


def _load_candidate(path, layer, base):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    meta = payload.get('metadata', {})
    if (meta.get('base_weight_sha256') != WEIGHT_SHA or meta.get('layer') != layer
            or meta.get('format') != 'spora-qwen-layer-replacement-candidate-v1'):
        raise ValueError(f'Candidate provenance mismatch for layer {layer}')
    module = SPORALayerReplacement(base.config.hidden_size, base.config.num_attention_heads,
                                  max_length=meta['max_length'], layer_idx=layer).to('cuda').eval()
    module.load_state_dict(payload['state'], strict=True)
    return module


def _swap(base, replacements, originals):
    for layer in LAYERS:
        base.model.layers[layer] = replacements[layer]


def _restore(base, originals):
    for layer in LAYERS:
        base.model.layers[layer] = originals[layer]


def _score(teacher_logits, student_logits):
    teacher = teacher_logits[:, :-1].float()
    student = student_logits[:, :-1].float()
    p = teacher.softmax(-1)
    kl = (p * (teacher.log_softmax(-1) - student.log_softmax(-1))).sum(-1)
    agree = (teacher.argmax(-1) == student.argmax(-1)).sum().item()
    count = teacher.shape[0] * teacher.shape[1]
    return {'agreement': int(agree), 'tokens': int(count), 'kl_sum': float(kl.sum())}


def _evaluate(integrated, base, replacements, originals, tokenizer, rows):
    _swap(base, replacements, originals)
    records = []
    try:
        with torch.inference_mode():
            for row in rows:
                inputs, controls = _inputs(tokenizer, row)
                logits = integrated(**inputs, **controls).logits
                teacher_logits = row['_teacher_logits'].to('cuda', dtype=torch.float32)
                records.append({'question': row['text'], 'intent': row['label'],
                                **_score(teacher_logits, logits)})
    finally:
        _restore(base, originals)
    count = sum(item['tokens'] for item in records)
    summary = {'top1_agreement': sum(item['agreement'] for item in records) / count,
               'mean_token_kl_nats': sum(item['kl_sum'] for item in records) / count,
               'scored_tokens': count}
    return summary, records


def run():
    global _organism, _state
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
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
    train_rows = examples(TRAIN)
    held_rows = examples(HELD)
    originals = {layer: base.model.layers[layer] for layer in LAYERS}
    active_candidates = ({layer: START_DIR / f'layer-{layer}.pt' for layer in LAYERS}
                         if START_DIR.exists() else START_CANDIDATES)
    replacements = {layer: _load_candidate(active_candidates[layer], layer, base) for layer in LAYERS}
    start_digests = {str(layer): sha(path) for layer, path in active_candidates.items()}

    # Teacher targets include the same native memory reads as the student.
    teacher_cache = []
    with torch.inference_mode():
        for row in train_rows + held_rows:
            inputs, controls = _inputs(tokenizer, row)
            teacher = integrated(**inputs, **controls, output_hidden_states=True)
            row['_teacher_logits'] = teacher.logits.detach().to('cpu', dtype=torch.float16)
            if row in train_rows:
                row['_teacher_h7'] = teacher.hidden_states[7].detach().to('cpu', dtype=torch.float16)
                row['_teacher_h13'] = teacher.hidden_states[13].detach().to('cpu', dtype=torch.float16)
        teacher_cache = [{'row': row, 'inputs': _inputs(tokenizer, row)[0],
                          'controls': _inputs(tokenizer, row)[1]} for row in train_rows]

    before, before_rows = _evaluate(integrated, base, replacements, originals, tokenizer, held_rows)
    parameters = [parameter for module in replacements.values() for parameter in module.parameters()]
    initial_params = {layer: {name: value.detach().clone() for name, value in module.state_dict().items()}
                      for layer, module in replacements.items()}
    optimizer = torch.optim.AdamW(parameters, lr=1e-6, weight_decay=0.01, foreach=False)
    order = list(range(len(teacher_cache)))
    rng = random.Random(SEED + 1)
    rng.shuffle(order)
    trace = []
    for module in replacements.values():
        module.train()
    for step in range(STEPS):
        item = teacher_cache[order[step % len(order)]]
        row = item['row']
        teacher_logits = row['_teacher_logits'][:, :-1].to('cuda', dtype=torch.float32)
        target_h7 = row['_teacher_h7'].to('cuda', dtype=torch.float32)
        target_h13 = row['_teacher_h13'].to('cuda', dtype=torch.float32)
        length = item['inputs']['input_ids'].shape[1]
        positions = torch.arange(length, device='cuda').unsqueeze(0)
        # The full forward is the optimization path; teacher hidden alignment
        # protects each replaced block's local transformation under memory.
        _swap(base, replacements, originals)
        output = integrated(**item['inputs'], **item['controls'], output_hidden_states=True)
        logits = output.logits[:, :-1].float()
        h7, h13 = output.hidden_states[7].float(), output.hidden_states[13].float()
        tlogp = (teacher_logits / 2).log_softmax(-1)
        tprob = tlogp.exp()
        slogp = (logits / 2).log_softmax(-1)
        kl = (tprob * (tlogp - slogp)).sum(-1).mean() * 4
        teacher_top1 = teacher_logits.argmax(-1)
        top1_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), teacher_top1.reshape(-1))
        mse7 = (h7 - target_h7).square().mean() / target_h7.square().mean().clamp_min(1e-8)
        mse13 = (h13 - target_h13).square().mean() / target_h13.square().mean().clamp_min(1e-8)
        cosine = (1 - F.cosine_similarity(h7, target_h7, dim=-1).mean()
                  + 1 - F.cosine_similarity(h13, target_h13, dim=-1).mean())
        loss = 0.05 * kl + 0.5 * top1_loss + 0.1 * (mse7 + mse13) + 0.01 * cosine
        if not bool(torch.isfinite(loss)):
            _restore(base, originals)
            raise ValueError('Nonfinite joint memory-conditioned distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        _restore(base, originals)
        if step in (0, 19, 39, 59, 79, 99, 119):
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'logit_kl': float(kl.detach()), 'top1_loss': float(top1_loss.detach()),
                          'relative_mse_layer7': float(mse7.detach()),
                          'relative_mse_layer13': float(mse13.detach())})
    for module in replacements.values():
        module.eval()
    after, after_rows = _evaluate(integrated, base, replacements, originals, tokenizer, held_rows)

    checkpoint_dir = OUTPUT
    checkpoint_dir.mkdir(parents=False, exist_ok=False)
    candidate_paths = {}
    for layer, module in replacements.items():
        destination = checkpoint_dir / f'layer-{layer}.pt'
        metadata = {'format': 'spora-qwen-layer-replacement-candidate-v1',
                    'base_weight_sha256': WEIGHT_SHA, 'layer': layer,
                    'width': base.config.hidden_size, 'heads': base.config.num_attention_heads,
                    'max_length': 128, 'seed': SEED, 'train_steps': STEPS,
                    'objective': 'joint_native_memory_conditioned_next_token_and_hidden_distill'}
        atomic_save({'metadata': metadata,
                     'state': {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}},
                    destination)
        candidate_paths[str(layer)] = {'path': destination.name, 'sha256': sha(destination)}
    report = {
        'status': 'joint_memory_conditioned_two_layer_spora_distillation',
        'base_weight_sha256': WEIGHT_SHA,
        'individual_checkpoint_sha256': sha(SOURCE_CHECKPOINT),
        'start_candidates_sha256': start_digests,
        'candidate_checkpoints': candidate_paths,
        'layers': list(LAYERS),
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
            'Twelve temporal training prompts and nine authored holdout prompts.',
            'One individual, one key, two values and one routed event history.',
            'The DeepSeek base, SPORA memory reader and write controller remain frozen.',
            'No generated answers, other memory tasks or broader language-domain checks included.',
        ],
    }
    with (OUTPUT / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'before': before, 'after': after}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
