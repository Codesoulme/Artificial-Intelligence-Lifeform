"""Distill one DeepSeek decoder layer into a SPORA CausalBlock candidate."""
import hashlib
import argparse
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
from spora.language_training import atomic_save
from spora.structure_transfer import SPORALayerReplacement

LAYER = 12
OUTPUT = ROOT / 'runs/deepseek-spora-block-distill-layer12-v2-20261111'
LOGIT_OUTPUT = ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-v2-20261111'
WARMSTART = ROOT / 'runs/deepseek-spora-block-distill-layer12-logit-warmstart-20261111/replacement.pt'
TRAIN_SOURCE = ROOT / 'data/chinese-foundations.json'
HELD_SOURCE = ROOT / 'data/conversation-development-v2.json'
SEED = 90217
TRAIN_STEPS = 720
LOGIT_STEPS = 160
MAX_LENGTH = 128


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _training_texts():
    with TRAIN_SOURCE.open(encoding='utf-8') as stream:
        data = json.load(stream)
    rows = []
    for dialogue in data['dialogues']:
        for user, assistant in dialogue['turns']:
            rows.append(({'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}))
    if len(rows) < 32:
        raise ValueError('Insufficient authored training pairs')
    random.Random(SEED).shuffle(rows)
    return rows[:48]


def _held_messages():
    with HELD_SOURCE.open(encoding='utf-8') as stream:
        data = json.load(stream)
    return [[{'role': 'user', 'content': turn}]
            for scenario in data['scenarios'] for turn in scenario['turns']]


def _encode(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                        return_tensors='pt')
    ids = ids[:, :MAX_LENGTH]
    if ids.shape[1] < 2:
        raise ValueError('Teacher text tokenized to fewer than two tokens')
    return ids.to('cuda')


@torch.inference_mode()
def _collect(base, tokenizer, messages, layer_index):
    rows = []
    for message in messages:
        ids = _encode(tokenizer, message)
        mask = torch.ones_like(ids)
        outputs = base.model(input_ids=ids, attention_mask=mask, use_cache=False,
                             output_hidden_states=True, return_dict=True)
        source = outputs.hidden_states[layer_index][0].detach().to(device='cpu', dtype=torch.float16)
        target = outputs.hidden_states[layer_index + 1][0].detach().to(device='cpu', dtype=torch.float16)
        rows.append((source, target))
    return rows


def _measure(block, examples):
    mse, cosine = [], []
    with torch.inference_mode():
        for source, target in examples:
            x = source.to('cuda', dtype=torch.float32).unsqueeze(0)
            y = target.to('cuda', dtype=torch.float32).unsqueeze(0)
            positions = torch.arange(x.shape[1], device='cuda').unsqueeze(0)
            prediction = block(x, position_ids=positions)
            mse.append(float((prediction - y).square().mean() / y.square().mean().clamp_min(1e-8)))
            cosine.append(float(F.cosine_similarity(prediction, y, dim=-1).mean()))
    return {'relative_mse': sum(mse) / len(mse), 'mean_cosine': sum(cosine) / len(cosine),
            'examples': len(examples)}


@torch.inference_mode()
def _full_model_agreement(base, block, tokenizer, messages, layer_index):
    layer = base.model.layers[layer_index]
    baseline_top1, candidate_top1, divergences = [], [], []
    try:
        for message in messages:
            ids = _encode(tokenizer, message)
            mask = torch.ones_like(ids)
            base.model.layers[layer_index] = layer
            teacher_logits = base(input_ids=ids, attention_mask=mask, use_cache=False,
                                  return_dict=True).logits[:, :-1].float()
            base.model.layers[layer_index] = block
            student_logits = base(input_ids=ids, attention_mask=mask, use_cache=False,
                                  return_dict=True).logits[:, :-1].float()
            p = teacher_logits.softmax(-1)
            log_p = teacher_logits.log_softmax(-1)
            log_q = student_logits.log_softmax(-1)
            divergences.append(float((p * (log_p - log_q)).sum(-1).mean()))
            baseline_top1.extend(teacher_logits.argmax(-1).flatten().tolist())
            candidate_top1.extend(student_logits.argmax(-1).flatten().tolist())
    finally:
        base.model.layers[layer_index] = layer
    return {
        'token_top1_agreement': sum(a == b for a, b in zip(baseline_top1, candidate_top1)) / len(baseline_top1),
        'mean_token_kl_nats': sum(divergences) / len(divergences),
        'held_sequences': len(messages),
        'scored_next_tokens': len(baseline_top1),
    }


def run(logit_distill=False, layer_index=LAYER):
    if type(layer_index) is not int or layer_index < 0:
        raise ValueError('Layer index must be a nonnegative integer')
    if logit_distill and layer_index != LAYER:
        raise ValueError('Logit warm-start experiment is currently bound to layer 12')
    if layer_index == LAYER:
        output_dir = LOGIT_OUTPUT if logit_distill else OUTPUT
    else:
        output_dir = ROOT / f'runs/deepseek-spora-block-distill-layer{layer_index}-20261111'
    if output_dir.exists():
        raise FileExistsError(output_dir)
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
    config = base.config
    if (config.model_type != 'qwen2' or config.hidden_size != 1536 or config.num_hidden_layers != 28
            or layer_index >= config.num_hidden_layers):
        raise ValueError('Unexpected DeepSeek/Qwen architecture')
    train_messages = _training_texts()
    held_messages = _held_messages()
    train_features = _collect(base, tokenizer, train_messages, layer_index)
    held_features = _collect(base, tokenizer, held_messages, layer_index)
    replacement = SPORALayerReplacement(config.hidden_size, config.num_attention_heads,
                                        max_length=MAX_LENGTH, layer_idx=layer_index).to('cuda').train()
    if logit_distill:
        warm = torch.load(WARMSTART, map_location='cpu', weights_only=True)
        if (warm.get('metadata', {}).get('base_weight_sha256') != WEIGHT_SHA
                or warm.get('metadata', {}).get('layer') != layer_index
                or warm.get('metadata', {}).get('format') != 'spora-qwen-layer-replacement-candidate-v1'):
            raise ValueError('Warm-start candidate provenance mismatch')
        replacement.load_state_dict(warm['state'], strict=True)
    initial = _measure(replacement, held_features)
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-5 if logit_distill else 7e-5,
                                  weight_decay=0.01, foreach=False)
    updates = []
    order_rng = random.Random(SEED + 1)
    order = list(range(len(train_features)))
    steps = LOGIT_STEPS if logit_distill else TRAIN_STEPS
    for step in range(steps):
        if step % len(order) == 0:
            order_rng.shuffle(order)
        row_index = order[step % len(order)]
        source, target = train_features[row_index]
        x = source.to('cuda', dtype=torch.float32).unsqueeze(0)
        y = target.to('cuda', dtype=torch.float32).unsqueeze(0)
        positions = torch.arange(x.shape[1], device='cuda').unsqueeze(0)
        prediction = replacement(x, position_ids=positions)
        relative_mse = (prediction - y).square().mean() / y.square().mean().clamp_min(1e-8)
        cosine_loss = 1 - F.cosine_similarity(prediction, y, dim=-1).mean()
        if logit_distill:
            ids = _encode(tokenizer, train_messages[row_index])
            mask = torch.ones_like(ids)
            layer = base.model.layers[layer_index]
            with torch.no_grad():
                teacher_logits = base(input_ids=ids, attention_mask=mask, use_cache=False,
                                      return_dict=True).logits[:, :-1].float()
            try:
                base.model.layers[layer_index] = replacement
                student_logits = base(input_ids=ids, attention_mask=mask, use_cache=False,
                                      return_dict=True).logits[:, :-1].float()
            finally:
                base.model.layers[layer_index] = layer
            temperature = 2.0
            teacher_log_probs = (teacher_logits / temperature).log_softmax(-1)
            teacher_probs = teacher_log_probs.exp()
            student_log_probs = (student_logits / temperature).log_softmax(-1)
            kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1).mean() * temperature**2
            loss = 0.05 * kl + relative_mse + 0.1 * cosine_loss
        else:
            kl = None
            loss = relative_mse + 0.1 * cosine_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite block distillation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(replacement.parameters(), 1.0)
        optimizer.step()
        checkpoints = ((0, 39, 79, 119, 159) if logit_distill else (0, 99, 239, 479, 719))
        if step in checkpoints:
            row = {'step': step + 1, 'loss': float(loss.detach()),
                   'relative_mse': float(relative_mse.detach()),
                   'mean_cosine': float(F.cosine_similarity(prediction.detach(), y, dim=-1).mean())}
            if kl is not None:
                row['logit_kl'] = float(kl.detach())
            updates.append(row)
    replacement.eval()
    held_hidden = _measure(replacement, held_features)
    full_agreement = _full_model_agreement(base, replacement, tokenizer, held_messages, layer_index)
    state = {name: tensor.detach().cpu() for name, tensor in replacement.state_dict().items()}
    checkpoint = output_dir / 'replacement.pt'
    output_dir.mkdir(parents=False, exist_ok=False)
    metadata = {
        'format': 'spora-qwen-layer-replacement-candidate-v1',
        'base_weight_sha256': WEIGHT_SHA,
        'base_config_sha256': _hash(MODEL / 'config.json'),
        'train_source_sha256': _hash(TRAIN_SOURCE),
        'held_source_sha256': _hash(HELD_SOURCE),
        'layer': layer_index,
        'width': config.hidden_size,
        'heads': config.num_attention_heads,
        'max_length': MAX_LENGTH,
        'seed': SEED,
        'train_steps': steps,
        'distillation_objective': 'next_token_kl_plus_hidden_alignment' if logit_distill else 'hidden_state_mse_cosine',
    }
    atomic_save({'metadata': metadata, 'state': state}, checkpoint)
    report = {
        'status': 'single_layer_spora_structure_distillation_candidate',
        **metadata,
        'train_sequences': len(train_features),
        'held_sequences': len(held_features),
        'initial_held_hidden': initial,
        'final_held_hidden': held_hidden,
        'full_model_agreement': full_agreement,
        'loss_trace': updates,
        'candidate_sha256': sha(checkpoint),
        'elapsed_seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'device': torch.cuda.get_device_name(0),
        'limitations': [
            'One distilled layer and one project-authored training source; no open-web corpus used.',
            'Held scenarios are project development material and not independent chat acceptance.',
            'The candidate replaces a layer only for full-sequence forward; KV-cache generation is rejected.',
            'No base weights or original SPORA individual are changed.',
            'A single block does not prove multi-layer replacement stability or retained free chat.',
        ],
    }
    with (output_dir / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'initial_held_hidden': initial, 'final_held_hidden': held_hidden,
                      'full_model_agreement': full_agreement}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logit-distill', action='store_true')
    parser.add_argument('--layer-index', type=int, default=LAYER)
    args = parser.parse_args()
    run(logit_distill=args.logit_distill, layer_index=args.layer_index)
