"""Bounded adapter training on paired synthetic matrices; no real memory writes."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.open_base_baseline import MODEL, INDIVIDUAL, INDIVIDUAL_SHA, WEIGHT_SHA, REVISION, sha
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks

OUTPUT = ROOT/'runs/open-base-synthetic-alignment-20261109'
TRAIN_PROMPTS = ('记忆里的编号是几？只回答1或2。', '请读取编号，只回复1或2。',
                 '这个状态对应哪个数字？仅回答1或2。', '回答记忆编号：1或2。')
HELD_PROMPTS = ('请说出状态里的编号，只写1或2。', '在1和2之间选择记忆编号。')
TRAIN_SEEDS = tuple(range(100, 108))
HELD_SEEDS = (700, 701, 702, 703)


def make_state(width, seed, device, swap=False, empty=False):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    matrices = []
    for _ in range(2):
        noise = torch.randn(2, width, width, generator=generator)*0.01
        values = torch.stack((torch.eye(width), -torch.eye(width))) + noise
        values = F.normalize(values.flatten(1), dim=1).reshape(2, width, width)
        if swap:
            values = values.flip(0)
        if empty:
            values.zero_()
        matrices.append(values.to(device))
    return SimpleNamespace(current=matrices, pages=[])


def encode_prompt(tokenizer, prompt, device):
    formatted = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                               add_generation_prompt=True)
    return tokenizer(formatted, return_tensors='pt').to(device)


def choice_ids(tokenizer):
    choices = [tokenizer.encode(value, add_special_tokens=False) for value in ('1', '2')]
    if any(len(row) != 1 for row in choices) or choices[0] == choices[1]:
        raise ValueError('Expected distinct single-token choices')
    return [row[0] for row in choices]


@torch.no_grad()
def evaluate(model, layers, adapter, tokenizer, choices, width, seeds, prompts, device, mode='normal'):
    pairs = correct = 0
    margins = []
    for seed in seeds:
        for prompt in prompts:
            state = make_state(width, seed, device, swap=mode == 'swapped', empty=mode == 'empty')
            inputs = encode_prompt(tokenizer, prompt, device)
            inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
            if mode == 'disabled':
                logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
            else:
                with memory_hooks(layers, adapter, state):
                    logits = model(**inputs, use_cache=False).logits[:, -1, choices].float()
            targets = torch.tensor([1, 0] if mode == 'swapped' else [0, 1], device=device)
            predictions = logits.argmax(-1)
            correct += int((predictions == targets).sum())
            pairs += int(bool((predictions == targets).all()))
            margins.extend((logits.gather(1, targets[:, None]).squeeze(1) -
                            logits.gather(1, (1-targets)[:, None]).squeeze(1)).tolist())
    examples = 2*len(seeds)*len(prompts)
    return {'examples': examples, 'correct': correct, 'pairs': examples//2, 'both_correct': pairs,
            'mean_correct_vs_other_logit': sum(margins)/len(margins)}


def run(output=OUTPUT, steps=40):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'runs').resolve()) or output == (ROOT/'runs').resolve():
        raise ValueError('Output must be a new directory under runs')
    if type(steps) is not int or not 1 <= steps <= 400:
        raise ValueError('Steps must be between 1 and 400')
    if output.exists():
        raise FileExistsError(output)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA or sha(INDIVIDUAL) != INDIVIDUAL_SHA:
        raise ValueError('Protected input mismatch')
    torch.manual_seed(23702)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    choices = choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='eager').to('cuda').eval()
    model.requires_grad_(False)
    organism, original_state, _ = load(INDIVIDUAL, 'cuda')
    before = frozen_digest(organism, original_state)
    width = organism.config.memory_width
    adapter = OpenBaseMemory(model.config.hidden_size, width).to('cuda')
    layers = [model.model.layers[5], model.model.layers[17]]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.)
    log = []
    for step in range(steps):
        state = make_state(width, TRAIN_SEEDS[step % len(TRAIN_SEEDS)], 'cuda')
        prompt = TRAIN_PROMPTS[(step // len(TRAIN_SEEDS)) % len(TRAIN_PROMPTS)]
        inputs = encode_prompt(tokenizer, prompt, 'cuda')
        inputs = {k: v.expand(2, -1) for k, v in inputs.items()}
        optimizer.zero_grad(set_to_none=True)
        with memory_hooks(layers, adapter, state):
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
            loss = F.cross_entropy(logits, torch.tensor(choices, device='cuda'))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite loss')
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step+1) % 50 == 0 or step+1 == steps:
            log.append({'step': step+1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
            print(json.dumps(log[-1]), flush=True)
    adapter.eval()
    results = {
        'train': evaluate(model, layers, adapter, tokenizer, choices, width, TRAIN_SEEDS, TRAIN_PROMPTS, 'cuda'),
        'new_states_new_prompts': evaluate(model, layers, adapter, tokenizer, choices, width, HELD_SEEDS, HELD_PROMPTS, 'cuda'),
        'swapped': evaluate(model, layers, adapter, tokenizer, choices, width, HELD_SEEDS, HELD_PROMPTS, 'cuda', 'swapped'),
        'empty': evaluate(model, layers, adapter, tokenizer, choices, width, HELD_SEEDS, HELD_PROMPTS, 'cuda', 'empty'),
        'disabled': evaluate(model, layers, adapter, tokenizer, choices, width, HELD_SEEDS, HELD_PROMPTS, 'cuda', 'disabled'),
    }
    if (any(p.grad is not None for p in model.parameters()) or before != frozen_digest(organism, original_state)
            or sha(INDIVIDUAL) != INDIVIDUAL_SHA or sha(MODEL/'model.safetensors') != WEIGHT_SHA):
        raise ValueError('Frozen source changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output/'adapter.pt'
    torch.save({'format': 'spora-open-base-synthetic-v1', 'state': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'layers': [5, 17]}, snapshot)
    report = {'status': 'synthetic_probe_not_real_memory_alignment', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
              'adapter_sha256': sha(snapshot), 'layers': [5, 17], 'steps': steps,
              'choice_ids': choices, 'train_seeds': TRAIN_SEEDS, 'held_seeds': HELD_SEEDS,
              'train_prompts': TRAIN_PROMPTS, 'held_prompts': HELD_PROMPTS, 'log': log, 'results': results,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter()-start,
              'limitations': ['Synthetic matrices encode a fixed two-class rule; not actual SPORA memory values.',
                              'Accuracy compares only logits for choices 1 and 2, not free generation.',
                              'New states share the same artificial rule and are not a test of arbitrary memories.',
                              'Frozen base and real individual were not updated; no data or external teacher model was used.']}
    with (output/'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output, args.steps)
