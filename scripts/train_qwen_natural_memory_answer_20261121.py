"""Train a frozen-Qwen memory reader on natural Chinese answer templates."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_addressed_full_answer import answer_batch, create_snapshots
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, MODEL, WEIGHT_SHA, REVISION, sha
from scripts.open_base_real_memory_alignment import HELD_PAIRS, TRAIN_PAIRS
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


TRAIN_PROMPTS = (
    '请从记忆中查一下对象{key}当前保存的数字。',
    '我想知道编号{key}现在记录的数值。',
    '对象{key}目前对应哪个数字？',
    '帮我读出对象{key}的当前数值。',
)
HELD_PROMPTS = (
    '我之前记下的对象{key}现在是多少？',
    '从记录里看，对象{key}当前数字是几？',
    '你记得编号{key}保存的数值吗？',
)
DEV_VALUES = (0, 1, 2, 3, 4, 7, 10, 15)


def addresses_for(organism, key):
    return [memory.current_basis[key:key + 1].detach() for memory in organism.memories]


@torch.inference_mode()
def generate_batch(model, tokenizer, question, state, addresses, layers, adapter):
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': question}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to('cuda')
    inputs = {key: value.expand(2, -1) for key, value in inputs.items()}
    with memory_hooks(layers, adapter, state, current_only=True, addresses=addresses):
        result = model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return [tokenizer.decode(row[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            for row in result], [bool(row[-1].item() == tokenizer.eos_token_id) for row in result]


def run(output, steps):
    output = output.resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve():
        raise ValueError('Output must be a new directory under runs')
    if output.exists() or type(steps) is not int or not 1 <= steps <= 2400:
        raise ValueError('Invalid steps or existing output')
    if sha(MODEL / 'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected source mismatch')
    started = time.perf_counter()
    torch.manual_seed(53112)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    snapshots, actions = create_snapshots(organism, original, TRAIN_PAIRS + HELD_PAIRS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation='eager',
    ).to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda')
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=.001, weight_decay=0.)
    tasks = [(pair, position, template, value)
             for pair in TRAIN_PAIRS for position in range(2)
             for template in TRAIN_PROMPTS for value in range(16)]
    generator = torch.Generator().manual_seed(53112)
    order = torch.randperm(len(tasks), generator=generator).tolist()
    losses = []
    for step in range(steps):
        if step and step % len(order) == 0:
            order = torch.randperm(len(tasks), generator=generator).tolist()
        pair, position, template, value = tasks[order[step % len(order)]]
        question = template.format(key=pair[position])
        answers = [value, (value + 1) % 16] if position == 0 else [(value + 1) % 16, value]
        inputs, labels = answer_batch(tokenizer, question, answers, 'cuda')
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks(layers, adapter, snapshots[(pair, value)], current_only=True,
                          addresses=addresses_for(organism, pair[position])):
            logits = model(**inputs, use_cache=False).logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            losses.append({'step': step + 1, 'loss': float(loss.detach()),
                           'gradient_norm': float(norm)})
            print(json.dumps(losses[-1]), flush=True)
    adapter.eval()
    rows = []
    for pair in HELD_PAIRS:
        for position, key in enumerate(pair):
            for template in HELD_PROMPTS:
                question = template.format(key=key)
                addresses = addresses_for(organism, key)
                answers, eos = generate_batch(
                    model, tokenizer, question, snapshots[(pair, DEV_VALUES[0])],
                    addresses, layers, adapter)
                expected = [DEV_VALUES[0], (DEV_VALUES[0] + 1) % 16] if position == 0 else [
                    (DEV_VALUES[0] + 1) % 16, DEV_VALUES[0]]
                rows.append({
                    'pair': list(pair), 'position': position, 'question': question,
                    'expected': expected, 'answers': answers, 'eos': eos,
                    'correct': [answer.rstrip('。.!！ ') == str(target)
                                for answer, target in zip(answers, expected, strict=True)],
                })
    if (any(parameter.grad is not None for parameter in model.parameters())
            or before != frozen_digest(organism, original)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA
            or sha(MODEL / 'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Protected source changed')
    output.mkdir(parents=False, exist_ok=False)
    adapter_path = output / 'adapter.pt'
    torch.save({
        'format': 'spora-qwen-natural-memory-answer-v1',
        'state': {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
        'layers': [5, 17], 'train_prompts': TRAIN_PROMPTS,
    }, adapter_path)
    report = {
        'status': 'natural_memory_answer_reader_candidate', 'revision': REVISION,
        'steps': steps, 'train_pairs': TRAIN_PAIRS, 'held_pairs': HELD_PAIRS,
        'train_prompts': TRAIN_PROMPTS, 'held_prompts': HELD_PROMPTS,
        'dev_values': DEV_VALUES, 'losses': losses, 'rows': rows,
        'correct_answers': sum(sum(row['correct']) for row in rows),
        'total_answers': sum(len(row['correct']) for row in rows),
        'both_correct': sum(all(row['correct']) for row in rows),
        'all_eos': sum(all(row['eos']) for row in rows),
        'adapter_sha256': sha(adapter_path), 'weight_sha256': WEIGHT_SHA,
        'individual_sha256': INDIVIDUAL_SHA, 'action_counts': {
            action: actions.count(action) for action in sorted(set(actions))},
        'seconds': time.perf_counter() - started,
        'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'The address is supplied directly by the controlled experiment; this does not test natural routing.',
            'Only one held value state is generated per question; training covers finite numeric values.',
            'This candidate is not connected to the formal runtime or individual.',
        ],
    }
    with (output / 'result.json').open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('correct_answers', 'total_answers', 'both_correct', 'all_eos')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1600)
    args = parser.parse_args()
    run(args.output, args.steps)
