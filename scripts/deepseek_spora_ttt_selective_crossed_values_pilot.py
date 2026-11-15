"""Train and audit a TTT reader against value-position-only writes.

The value boundary is supplied by the experiment so this is an oracle
diagnostic, not an autonomous memory writer.  It tests whether the current
whole-fact write target is masking the value signal before the reader sees it.
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_selective_write_audit import selected_memory
from spora.research_ttt_ntp import attach_read_adapters, install, use_memory


OUT = ROOT / 'runs/deepseek-spora-ttt-selective-crossed-values-20261116'
LAYERS = (6, 12)
RANK = 16
STEPS = 384
LR = 1e-3
ETA = pilot.ETA
RIDGE = pilot.RIDGE
SEED = 73119
ALT_TEMPLATE = '请说出{name}偏爱的饮品名称。'


def exact(answer, value):
    return answer.strip().rstrip('。.!！ ').strip() == value


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, LAYERS)
    adapters = attach_read_adapters(wrappers, rank=RANK)
    model.requires_grad_(False)
    parameters = [parameter for adapter in adapters.values() for parameter in adapter.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)

    memories = {}
    for name in crossed.TRAIN_NAMES + crossed.HELD_NAMES:
        for value in crossed.TRAIN_VALUES + crossed.HELD_VALUES:
            memories[(name, value)] = selected_memory(
                model, tokenizer, wrappers, f'{name}喜欢喝{value}。', value)[0]
        print(json.dumps({'prepared': name, 'states': len(memories)}, ensure_ascii=False), flush=True)

    train = []
    for name in crossed.TRAIN_NAMES:
        for value in crossed.TRAIN_VALUES:
            for template in crossed.TEMPLATES:
                question = template.format(name=name)
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                ids = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
                target = torch.tensor([ids], device='cuda', dtype=torch.long)
                train.append({'name': name, 'value': value, 'question': question,
                              'prompt': prompt,
                              'ids': torch.cat((prompt['input_ids'], target), dim=1)})

    def generate(question, memory=None):
        return pilot.generate(model, tokenizer, question, memory)

    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [generate(question) for question in ordinary]
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=1e-4, foreach=False)
    order = list(range(len(train)))
    trace = []
    for step in range(STEPS):
        if step % len(order) == 0:
            random.shuffle(order)
        item = train[order[step % len(order)]]
        memory = memories[(item['name'], item['value'])]
        prefix = item['prompt']['input_ids'].shape[1]
        target = item['ids'][:, prefix:]
        optimizer.zero_grad(set_to_none=True)
        with use_memory(memory):
            result = model(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                           use_cache=False, return_dict=True)
        scores = result.logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
        loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                                 target.reshape(-1))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({'step': step + 1, 'loss': float(loss.detach()),
                      'gradient_norm': float(gradient)})
        if (step + 1) % 32 == 0:
            print(json.dumps(trace[-1]), flush=True)

    rows = []
    groups = (
        ('train_pair', crossed.TRAIN_NAMES, crossed.TRAIN_VALUES, crossed.TEMPLATES[0]),
        ('held_name', crossed.HELD_NAMES, crossed.TRAIN_VALUES, crossed.TEMPLATES[0]),
        ('held_value', crossed.HELD_NAMES, crossed.HELD_VALUES, crossed.TEMPLATES[0]),
        ('held_question', crossed.HELD_NAMES, crossed.TRAIN_VALUES, ALT_TEMPLATE),
    )
    for group, names, values, template in groups:
        for name in names:
            question = template.format(name=name)
            baseline = generate(question)
            for value in values:
                answer = generate(question, memories[(name, value)])
                rows.append({'group': group, 'name': name, 'value': value,
                             'question': question, 'answer': answer,
                             'no_memory': baseline,
                             'exact': exact(answer['answer'], value),
                             'baseline_exact': exact(baseline['answer'], value)})
            print(json.dumps({'evaluated': group, 'name': name}, ensure_ascii=False), flush=True)

    ordinary_after = [generate(question) for question in ordinary]
    if ordinary_before != ordinary_after:
        raise ValueError('Memory-off ordinary control changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')

    summary = {}
    for group, names, _, _ in groups:
        subset = [row for row in rows if row['group'] == group]
        summary[group] = {
            'cases': len(subset),
            'exact': sum(row['exact'] for row in subset),
            'baseline_exact': sum(row['baseline_exact'] for row in subset),
            'all_values_correct': sum(all(row['exact'] for row in subset if row['name'] == name)
                                      for name in names),
            'names': len(names),
        }

    OUT.mkdir()
    payload = {
        'format': 'spora-ttt-selective-crossed-values-v1',
        'base_sha256': base_sha, 'layers': list(LAYERS), 'rank': RANK,
        'eta': ETA, 'ridge': RIDGE, 'steps': STEPS, 'learning_rate': LR, 'seed': SEED,
        'adapters': {str(layer): {name: tensor.detach().cpu()
                                  for name, tensor in adapter.state_dict().items()}
                     for layer, adapter in adapters.items()},
    }
    torch.save(payload, OUT / 'adapters.pt')
    adapter_sha = p.sha(OUT / 'adapters.pt')
    report = {
        'status': 'ttt_selective_crossed_values_pilot', 'summary': summary,
        'rows': rows, 'trace': trace, 'train_names': crossed.TRAIN_NAMES,
        'held_names': crossed.HELD_NAMES, 'train_values': crossed.TRAIN_VALUES,
        'held_values': crossed.HELD_VALUES, 'base_sha256': base_sha,
        'adapter_sha256': adapter_sha, 'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Value boundaries are supplied by the experiment; no learned writer is evaluated.',
            'The adapter is trained and evaluated on synthetic one-fact beverage records.',
            'The selected-write distribution differs from the whole-fact candidate.',
            'No persistence, natural routing, correction, or open-domain dialogue is tested.',
        ],
    }
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
