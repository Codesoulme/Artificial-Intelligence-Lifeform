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
    held_rows = questions(range(12), HELD_TEMPLATES)
    held_features = final_features(
        model, tokenizer, [question for question, _ in held_rows])
    held_gate_features = gate_module.hidden_features(
        model, tokenizer, [question for question, _ in held_rows])
    with torch.inference_mode():
        train_tens, train_ones = address_head(train_features)
        train_predicted = train_tens.argmax(-1) * 10 + train_ones.argmax(-1)
        train_accuracy = int((train_predicted
                              == torch.tensor([key for _, key in train_rows], device='cuda')).sum())
    with torch.inference_mode():
        held_tens, held_ones = address_head(held_features)
        tens_scores = held_tens.softmax(-1)
        ones_scores = held_ones.softmax(-1)
        predicted = (held_tens.argmax(-1) * 10 + held_ones.argmax(-1)).cpu().tolist()
        confidence = torch.minimum(tens_scores.max(-1).values,
                                   ones_scores.max(-1).values).cpu().tolist()

    gate_payload = torch.load(gate_path, map_location='cpu', weights_only=True)
    if gate_payload.get('format') != gate_module.IntentGate.format:
        raise ValueError('Unexpected intent gate format')
    intent_gate = gate_module.IntentGate(train_features.shape[1], gate_payload['cutoff']).cuda()
    intent_gate.load_state_dict(gate_payload['state'], strict=True)
    intent_gate.eval().requires_grad_(False)
    runtime_gate = gate_module.make_runtime_gate(intent_gate, gate_payload['cutoff'])

    rows = []
    for index, ((question, expected), predicted_key, score) in enumerate(
            zip(held_rows, predicted, confidence, strict=True)):
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
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump({
            'format': 'spora-qwen-natural-memory-address-diagnostic-v1',
            'steps': steps, 'learning_rate': learning_rate,
            'train_rows': len(train_rows), 'held_rows': len(held_rows),
            'train_accuracy': train_accuracy,
            'address_correct': sum(row['predicted_key'] == row['expected_key'] for row in rows),
            'memory_enabled': sum(row['memory_enabled'] for row in rows),
            'value_correct': sum(row['value_correct'] for row in rows),
            'losses': losses, 'rows': rows,
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
    print({'address_correct': sum(row['predicted_key'] == row['expected_key'] for row in rows),
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
