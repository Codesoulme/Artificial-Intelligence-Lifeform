"""Isolated online low-rank weight editing using observed next-token loss.

The observed value's token boundary is supplied by the experiment. No answer
question is used as a training input. This is an editing baseline, not an
autonomous memory writer or a deployed SPORA individual.
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-online-lora-memory-pilot-20261114'
LAYERS = (18, 23)
RANK = 8
STEPS = 32
LR = 0.002
EVENTS = (('小林', '酸奶'), ('阿宁', '豆浆'), ('小林', '咖啡'))
SEED = 48201


class OnlineDelta(nn.Module):
    """Trainable low-rank residual beside a frozen pretrained projection."""
    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError('A linear base projection is required')
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32, device=base.weight.device)
        nn.init.normal_(self.down.weight, std=base.in_features ** -0.5)
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    def forward(self, hidden):
        output = self.base(hidden)
        if not self.enabled:
            return output
        delta = self.up(self.down(hidden.float()))
        return output + delta.to(output.dtype)


def observation_example(tokenizer, name, value):
    """Form a causal text prefix and target from a direct observation."""
    fact = f'{name}喜欢喝{value}。'
    encoding = tokenizer(fact, return_offsets_mapping=True, add_special_tokens=False)
    start = fact.index(value)
    selected = [i for i, (left, right) in enumerate(encoding['offset_mapping'])
                if left < start + len(value) and right > start]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)) or selected[0] < 1:
        raise ValueError('Could not align a nonempty contiguous value span')
    token_ids = encoding['input_ids']
    return fact, token_ids[:selected[0]], token_ids[selected[0]:selected[-1] + 1]


def first_token_loss(model, prefix, target):
    input_ids = torch.tensor([prefix], dtype=torch.long, device='cuda')
    output = model(input_ids=input_ids, use_cache=False, logits_to_keep=1, return_dict=True)
    return F.cross_entropy(output.logits[:, -1].float(), torch.tensor([target], device='cuda'))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weights')
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in LAYERS:
        mlp = model.model.layers[layer].mlp
        module = OnlineDelta(mlp.down_proj, RANK)
        mlp.down_proj = module
        modules.append(module)
    parameters = [param for module in modules for projection in (module.down, module.up)
                  for param in projection.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=0, foreach=False)
    questions = ('请只回答小林喜欢喝什么饮料。', '请只回答阿宁喜欢喝什么饮料。',
                 '请写一句鼓励学习的话。')
    before = [pilot.generate(model, tokenizer, question) for question in questions]
    results = []
    for event, (name, value) in enumerate(EVENTS):
        fact, prefix, targets = observation_example(tokenizer, name, value)
        trace = []
        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            # Each observed next token is supervised after its true predecessors.
            offset = step % len(targets)
            loss = first_token_loss(model, prefix + targets[:offset], targets[offset])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite online-edit loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step in (0, STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        answers = [pilot.generate(model, tokenizer, question) for question in questions]
        results.append({'event': event, 'fact': fact, 'value_token_ids': targets,
                        'trace': trace, 'answers': answers})
        print(json.dumps({'event': event, 'fact': fact, 'trace': trace,
                          'answers': [row['answer'] for row in answers]}, ensure_ascii=False), flush=True)
    for module in modules:
        module.enabled = False
    disabled = [pilot.generate(model, tokenizer, question) for question in questions]
    if before != disabled or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Frozen base or disabled path changed')
    OUT.mkdir()
    torch.save({'format': 'online-lora-memory-pilot-v1', 'base_sha256': base_sha,
                'layers': LAYERS, 'rank': RANK,
                'adapters': [{key: tensor.detach().cpu() for key, tensor in module.state_dict().items()
                              if key.startswith(('down.', 'up.'))} for module in modules]}, OUT / 'adapter.pt')
    report = {'status': 'online_lora_observation_pilot', 'questions': questions, 'before': before,
              'events': results, 'disabled_matches_before': True, 'base_sha256': base_sha,
              'adapter_sha256': p.sha(OUT / 'adapter.pt'), 'steps_per_event': STEPS, 'lr': LR,
              'layers': LAYERS, 'rank': RANK, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The experiment supplies value-token boundaries and direct observations.',
                              'Only three authored events and three questions; no independent benchmark.',
                              'Low-rank weight updates are per-experiment and do not update an existing SPORA individual.',
                              'No historical question, crash recovery, or long-term interference tested.',
                              'Value-token prediction is supervised; it does not train a natural write decision.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'disabled_matches_before': True, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
