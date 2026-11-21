"""Train a Qwen-side natural-language address head for SPORA memory."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_qwen_memory_intent_gate_20261121 as gate_module
from scripts.open_base_baseline import INDIVIDUAL, MODEL, INDIVIDUAL_SHA, WEIGHT_SHA, sha
from scripts.open_base_real_memory_alignment import HELD_PAIRS, TRAIN_PAIRS, variants
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory


ADAPTER_PATH = ROOT / 'runs/open-base-addressed-full-answer-20261109/adapter.pt'
GATE_DIR = ROOT / 'runs/qwen-memory-intent-gate-v9-20261121'
TRAIN_TEMPLATES = (
    '请从记忆中查一下对象{key}当前保存的数字。',
    '我想知道编号{key}现在记录的数值。',
    '对象{key}目前对应哪个数字？',
    '帮我读出对象{key}的当前数值。',
)
HELD_TEMPLATES = (
    '我之前记下的对象{key}现在是多少？',
    '从记录里看，对象{key}当前数字是几？',
    '你记得编号{key}保存的数值吗？',
)


class AddressHead(nn.Module):
    format = 'spora-qwen-natural-memory-address-v1'

    def __init__(self, feature_width, classes=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(feature_width, 256), nn.GELU())
        self.tens = nn.Linear(256, 10)
        self.ones = nn.Linear(256, 10)

    def forward(self, features):
        hidden = self.trunk(features)
        return self.tens(hidden), self.ones(hidden)


class TokenAddressHead(nn.Module):
    format = 'spora-qwen-natural-memory-token-address-v1'

    def __init__(self, feature_width):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(feature_width + 2),
            nn.Linear(feature_width + 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_width * 3),
            nn.Linear(feature_width * 3, 256),
            nn.GELU(),
        )
        self.tens = nn.Linear(256, 10)
        self.ones = nn.Linear(256, 10)

    def forward(self, token_features, mask, attention_mask=None):
        batch, length, width = token_features.shape
        if attention_mask is None:
            attention_mask = mask
        if attention_mask.shape != mask.shape or not bool(attention_mask.any(dim=1).all()):
            raise ValueError('Attention mask must be nonempty and match the token mask')
        positions = torch.arange(length, device=token_features.device,
                                 dtype=token_features.dtype).view(1, length, 1)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1).to(token_features.dtype)
        relative = positions / lengths.unsqueeze(-1)
        position_features = torch.cat((relative, 1. - relative), dim=-1)
        scores = self.score(torch.cat((token_features, position_features), dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        pooled = (weights.unsqueeze(-1) * token_features).sum(dim=1)
        last_indices = mask.sum(dim=1).long().clamp_min(1) - 1
        last = token_features[torch.arange(batch, device=token_features.device), last_indices]
        masked_mean = (token_features * mask.unsqueeze(-1)).sum(dim=1) / lengths
        hidden = self.trunk(torch.cat((pooled, last, masked_mean), dim=-1))
        return self.tens(hidden), self.ones(hidden), weights


def questions(keys, templates):
    return [(template.format(key=key), key) for key in keys for template in templates]


@torch.inference_mode()
def final_features(model, tokenizer, items):
    rows = []
    for question in items:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
        hidden = model(**inputs, use_cache=False, output_hidden_states=True).hidden_states[-1][0].float()
        rows.append(torch.cat((hidden[-1], hidden.mean(dim=0)), dim=0) / 2**.5)
    return torch.stack(rows).cuda()


@torch.inference_mode()
def final_token_features(model, tokenizer, items):
    rows = []
    for question in items:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
        hidden = model(**inputs, use_cache=False, output_hidden_states=True).hidden_states[-1][0].float()
        rows.append(hidden)
    return rows


@torch.inference_mode()
def content_token_features(model, tokenizer, items):
    rows = []
    for question in items:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
        hidden = model(**inputs, use_cache=False, output_hidden_states=True).hidden_states[-1][0].float()
        content_ids = tokenizer(question, add_special_tokens=False)['input_ids']
        full_ids = inputs.input_ids[0].tolist()
        matches = [start for start in range(len(full_ids) - len(content_ids) + 1)
                   if full_ids[start:start + len(content_ids)] == content_ids]
        if len(matches) != 1:
            raise ValueError(f'Could not uniquely locate user content in chat template: {question!r}')
        start = matches[0]
        rows.append(hidden[start:start + len(content_ids)])
    return rows


@torch.inference_mode()
def contextual_content_features(model, tokenizer, items):
    rows, attention_rows = [], []
    for question in items:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
        hidden = model(**inputs, use_cache=False, output_hidden_states=True).hidden_states[-1][0].float()
        content_ids = tokenizer(question, add_special_tokens=False)['input_ids']
        full_ids = inputs.input_ids[0].tolist()
        matches = [start for start in range(len(full_ids) - len(content_ids) + 1)
                   if full_ids[start:start + len(content_ids)] == content_ids]
        if len(matches) != 1:
            raise ValueError(f'Could not uniquely locate user content in chat template: {question!r}')
        start = matches[0]
        attention = torch.zeros(hidden.shape[0], dtype=torch.bool)
        attention[start:start + len(content_ids)] = True
        rows.append(hidden)
        attention_rows.append(attention)
    return rows, attention_rows


def train_address(features, targets, steps, learning_rate):
    head = AddressHead(features.shape[1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.)
    target = torch.tensor(targets, device='cuda', dtype=torch.long)
    losses = []
    for step in range(steps):
        tens, ones = head(features)
        loss = (nn.functional.cross_entropy(tens, target // 10)
                + nn.functional.cross_entropy(ones, target % 10))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    return head.eval().requires_grad_(False), losses


def pad_token_features(rows):
    width = rows[0].shape[-1]
    length = max(row.shape[0] for row in rows)
    features = torch.zeros(len(rows), length, width, device='cuda')
    mask = torch.zeros(len(rows), length, device='cuda', dtype=torch.bool)
    for index, row in enumerate(rows):
        size = row.shape[0]
        features[index, :size] = row
        mask[index, :size] = True
    return features, mask


def pad_token_masks(rows):
    length = max(row.shape[0] for row in rows)
    masks = torch.zeros(len(rows), length, device='cuda', dtype=torch.bool)
    for index, row in enumerate(rows):
        masks[index, :row.shape[0]] = row.to(device='cuda')
    return masks


def train_token_address(rows, targets, steps, learning_rate):
    features, mask = pad_token_features(rows)
    head = TokenAddressHead(features.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.)
    target = torch.tensor(targets, device='cuda', dtype=torch.long)
    losses = []
    for step in range(steps):
        tens, ones, _ = head(features, mask)
        loss = (nn.functional.cross_entropy(tens, target // 10)
                + nn.functional.cross_entropy(ones, target % 10))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    return head.eval().requires_grad_(False), losses, features, mask


def train_contextual_token_address(rows, attention_rows, targets, steps, learning_rate):
    features, mask = pad_token_features(rows)
    attention_mask = pad_token_masks(attention_rows)
    head = TokenAddressHead(features.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.)
    target = torch.tensor(targets, device='cuda', dtype=torch.long)
    losses = []
    for step in range(steps):
        tens, ones, _ = head(features, mask, attention_mask)
        loss = (nn.functional.cross_entropy(tens, target // 10)
                + nn.functional.cross_entropy(ones, target % 10))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    return head.eval().requires_grad_(False), losses, features, mask, attention_mask


def address_for(organism, key):
    if type(key) is not int or not 0 <= key < organism.config.key_count:
        raise ValueError('Predicted address is outside the individual key range')
    return [memory.current_basis[key:key + 1].detach() for memory in organism.memories]


def main(output, steps, learning_rate):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    gate_path = GATE_DIR / 'intent_gate.pt'
    sources = (MODEL / 'model.safetensors', MODEL / 'config.json', MODEL / 'tokenizer.json',
               MODEL / 'tokenizer_config.json', INDIVIDUAL, ADAPTER_PATH,
               gate_path, GATE_DIR / 'result.json')
    protected = {path: sha(path) for path in sources}
    started = time.perf_counter()
    torch.manual_seed(53111)
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
    adapter_payload = torch.load(ADAPTER_PATH, map_location='cpu', weights_only=True)
    if adapter_payload.get('format') != 'spora-open-base-p26-full-answer-v1':
        raise ValueError('Unexpected memory adapter format')
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    adapter.load_state_dict(adapter_payload['state'], strict=True)
    adapter.eval()
    layers = [model.model.layers[5], model.model.layers[17]]

    train_rows = questions(range(12), TRAIN_TEMPLATES)
    train_features = final_features(
        model, tokenizer, [question for question, _ in train_rows]).clone()
    address_head, losses = train_address(
        train_features, [key for _, key in train_rows], steps, learning_rate)
    train_token_rows = final_token_features(
        model, tokenizer, [question for question, _ in train_rows])
    token_head, token_losses, train_token_features, train_token_mask = train_token_address(
        train_token_rows, [key for _, key in train_rows], steps, learning_rate)
    train_content_rows = content_token_features(
        model, tokenizer, [question for question, _ in train_rows])
    content_head, content_losses, train_content_features, train_content_mask = train_token_address(
        train_content_rows, [key for _, key in train_rows], steps, learning_rate)
    train_context_rows, train_context_attention_rows = contextual_content_features(
        model, tokenizer, [question for question, _ in train_rows])
    context_head, context_losses, train_context_features, train_context_mask, train_context_attention = train_contextual_token_address(
        train_context_rows, train_context_attention_rows,
        [key for _, key in train_rows], steps, learning_rate)
    held_rows = questions(range(12), HELD_TEMPLATES)
    held_features = final_features(
        model, tokenizer, [question for question, _ in held_rows])
    held_token_rows = final_token_features(
        model, tokenizer, [question for question, _ in held_rows])
    held_token_features, held_token_mask = pad_token_features(held_token_rows)
    held_content_rows = content_token_features(
        model, tokenizer, [question for question, _ in held_rows])
    held_content_features, held_content_mask = pad_token_features(held_content_rows)
    held_context_rows, held_context_attention_rows = contextual_content_features(
        model, tokenizer, [question for question, _ in held_rows])
    held_context_features, held_context_mask = pad_token_features(held_context_rows)
    held_context_attention = pad_token_masks(held_context_attention_rows)
    held_gate_features = gate_module.hidden_features(
        model, tokenizer, [question for question, _ in held_rows])
    with torch.inference_mode():
        train_tens, train_ones = address_head(train_features)
        train_predicted = train_tens.argmax(-1) * 10 + train_ones.argmax(-1)
        train_accuracy = int((train_predicted
                              == torch.tensor([key for _, key in train_rows], device='cuda')).sum())
        train_token_tens, train_token_ones, _ = token_head(
            train_token_features, train_token_mask)
        train_token_predicted = train_token_tens.argmax(-1) * 10 + train_token_ones.argmax(-1)
        train_token_accuracy = int((train_token_predicted
                                    == torch.tensor([key for _, key in train_rows], device='cuda')).sum())
        train_content_tens, train_content_ones, _ = content_head(
            train_content_features, train_content_mask)
        train_content_predicted = train_content_tens.argmax(-1) * 10 + train_content_ones.argmax(-1)
        train_content_accuracy = int((train_content_predicted
                                     == torch.tensor([key for _, key in train_rows], device='cuda')).sum())
        train_context_tens, train_context_ones, _ = context_head(
            train_context_features, train_context_mask, train_context_attention)
        train_context_predicted = train_context_tens.argmax(-1) * 10 + train_context_ones.argmax(-1)
        train_context_accuracy = int((train_context_predicted
                                      == torch.tensor([key for _, key in train_rows], device='cuda')).sum())
    with torch.inference_mode():
        held_tens, held_ones = address_head(held_features)
        tens_scores = held_tens.softmax(-1)
        ones_scores = held_ones.softmax(-1)
        baseline_predicted = (held_tens.argmax(-1) * 10 + held_ones.argmax(-1)).cpu().tolist()
        baseline_confidence = torch.minimum(tens_scores.max(-1).values,
                                            ones_scores.max(-1).values).cpu().tolist()
        token_tens, token_ones, token_weights = token_head(
            held_token_features, held_token_mask)
        token_tens_scores = token_tens.softmax(-1)
        token_ones_scores = token_ones.softmax(-1)
        token_predicted = (token_tens.argmax(-1) * 10 + token_ones.argmax(-1)).cpu().tolist()
        token_confidence = torch.minimum(token_tens_scores.max(-1).values,
                                         token_ones_scores.max(-1).values).cpu().tolist()
        token_focus = token_weights.argmax(-1).cpu().tolist()
        content_tens, content_ones, content_weights = content_head(
            held_content_features, held_content_mask)
        content_tens_scores = content_tens.softmax(-1)
        content_ones_scores = content_ones.softmax(-1)
        content_predicted = (content_tens.argmax(-1) * 10 + content_ones.argmax(-1)).cpu().tolist()
        content_confidence = torch.minimum(content_tens_scores.max(-1).values,
                                            content_ones_scores.max(-1).values).cpu().tolist()
        content_focus = content_weights.argmax(-1).cpu().tolist()
        context_tens, context_ones, context_weights = context_head(
            held_context_features, held_context_mask, held_context_attention)
        context_tens_scores = context_tens.softmax(-1)
        context_ones_scores = context_ones.softmax(-1)
        predicted = (context_tens.argmax(-1) * 10 + context_ones.argmax(-1)).cpu().tolist()
        confidence = torch.minimum(context_tens_scores.max(-1).values,
                                   context_ones_scores.max(-1).values).cpu().tolist()
        context_focus = context_weights.argmax(-1).cpu().tolist()

    gate_payload = torch.load(gate_path, map_location='cpu', weights_only=True)
    if gate_payload.get('format') != gate_module.IntentGate.format:
        raise ValueError('Unexpected intent gate format')
    intent_gate = gate_module.IntentGate(train_features.shape[1], gate_payload['cutoff']).cuda()
    intent_gate.load_state_dict(gate_payload['state'], strict=True)
    intent_gate.eval().requires_grad_(False)
    runtime_gate = gate_module.make_runtime_gate(intent_gate, gate_payload['cutoff'])

    rows = []
    for index, ((question, expected), predicted_key, score, baseline_key, baseline_score,
                token_key, token_score, content_key, content_score) in enumerate(
            zip(held_rows, predicted, confidence, baseline_predicted, baseline_confidence,
                token_predicted, token_confidence, content_predicted, content_confidence,
                strict=True)):
        pair = next(pair for pair in (*TRAIN_PAIRS, *HELD_PAIRS) if expected in pair)
        state = states[pair][0]
        generated, eos = gate_module.generate(
            model, tokenizer, question, adapter=adapter, state=state,
            addresses=address_for(organism, predicted_key), layers=layers,
            gate=runtime_gate, limit=8,
        )
        gate_score = intent_gate(held_gate_features[index:index + 1]).sigmoid().item()
        rows.append({
            'question': question, 'expected_key': expected, 'predicted_key': predicted_key,
            'address_confidence': score, 'memory_score': gate_score,
            'baseline_predicted_key': baseline_key,
            'baseline_address_confidence': baseline_score,
            'token_predicted_key': token_key,
            'token_address_confidence': token_score,
            'content_predicted_key': content_key,
            'content_address_confidence': content_score,
            'token_focus_index': token_focus[index],
            'content_focus_index': content_focus[index],
            'context_focus_index': context_focus[index],
            'memory_enabled': gate_score >= gate_payload['cutoff'],
            'answer': generated, 'eos': eos,
            'value_correct': generated.strip().strip('。') in ('1', '2') and
            generated.strip().strip('。') == str(1 if expected % 2 == 0 else 2),
        })

    if (before != frozen_digest(organism, original)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL / 'model.safetensors') != WEIGHT_SHA
            or any(sha(path) != digest for path, digest in protected.items())):
        raise ValueError('Protected input changed')
    torch.save({
        'format': AddressHead.format, 'model_weight_sha256': WEIGHT_SHA,
        'gate_sha256': protected[gate_path], 'state': address_head.state_dict(),
    }, output / 'address_head.pt')
    torch.save({
        'format': TokenAddressHead.format, 'model_weight_sha256': WEIGHT_SHA,
        'gate_sha256': protected[gate_path], 'state': token_head.state_dict(),
    }, output / 'token_address_head.pt')
    torch.save({
        'format': TokenAddressHead.format, 'model_weight_sha256': WEIGHT_SHA,
        'gate_sha256': protected[gate_path], 'state': content_head.state_dict(),
    }, output / 'content_address_head.pt')
    torch.save({
        'format': TokenAddressHead.format, 'model_weight_sha256': WEIGHT_SHA,
        'gate_sha256': protected[gate_path], 'state': context_head.state_dict(),
        'attention_scope': 'user_content_with_full_context',
    }, output / 'context_address_head.pt')
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump({
            'format': 'spora-qwen-natural-memory-address-diagnostic-v1',
            'steps': steps, 'learning_rate': learning_rate,
            'train_rows': len(train_rows), 'held_rows': len(held_rows),
            'train_accuracy': train_accuracy,
            'train_token_accuracy': train_token_accuracy,
            'train_content_accuracy': train_content_accuracy,
            'train_context_accuracy': train_context_accuracy,
            'baseline_address_correct': sum(row['baseline_predicted_key'] == row['expected_key'] for row in rows),
            'token_address_correct': sum(row['token_predicted_key'] == row['expected_key'] for row in rows),
            'content_address_correct': sum(row['content_predicted_key'] == row['expected_key'] for row in rows),
            'address_correct': sum(row['predicted_key'] == row['expected_key'] for row in rows),
            'memory_enabled': sum(row['memory_enabled'] for row in rows),
            'value_correct': sum(row['value_correct'] for row in rows),
            'losses': losses, 'token_losses': token_losses,
            'content_losses': content_losses, 'context_losses': context_losses, 'rows': rows,
            'sources': {str(path): digest for path, digest in protected.items()},
            'seconds': time.perf_counter() - started,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
            'limitations': [
                'The address head uses a finite controlled key set and synthetic natural-language templates.',
                'The value reader remains the finite numbered-task adapter.',
                'This candidate is not installed into the formal runtime or individual.',
            ],
        }, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({'baseline_address_correct': sum(row['baseline_predicted_key'] == row['expected_key'] for row in rows),
           'token_address_correct': sum(row['token_predicted_key'] == row['expected_key'] for row in rows),
           'content_address_correct': sum(row['content_predicted_key'] == row['expected_key'] for row in rows),
           'address_correct': sum(row['predicted_key'] == row['expected_key'] for row in rows),
           'memory_enabled': sum(row['memory_enabled'] for row in rows),
           'value_correct': sum(row['value_correct'] for row in rows)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--learning-rate', type=float, default=.005)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve():
        parser.error('Output must be a new directory under runs')
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error('Invalid training arguments')
    main(output, args.steps, args.learning_rate)
