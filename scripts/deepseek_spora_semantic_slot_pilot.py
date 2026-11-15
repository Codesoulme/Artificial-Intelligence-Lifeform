"""Validate a low-dimensional semantic slot against unseen value tokens."""
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
from spora.research_semantic_slot import install, use_slot


OUT = ROOT / 'runs/deepseek-spora-semantic-slot-pilot-20261116'
RANK = 16
STEPS = 384
LR = 1e-3
SEED = 63271
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
    bridge = install(model, rank=RANK)
    model.requires_grad_(False)
    bridge.requires_grad_(True)
    trainable = list(bridge.parameters())

    def slot(value):
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        if not token_ids:
            raise ValueError('Value has no token')
        return model.model.embed_tokens.weight[token_ids[0]].detach().float()

    def generate(question, memory_slot=None):
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        kwargs = dict(**prompt, use_cache=True, max_new_tokens=24, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        with torch.inference_mode():
            if memory_slot is None:
                output = model.generate(**kwargs)
            else:
                with use_slot(memory_slot):
                    output = model.generate(**kwargs)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
                'token_ids': tail}

    train = []
    for name in crossed.TRAIN_NAMES:
        for value in crossed.TRAIN_VALUES:
            for template in crossed.TEMPLATES:
                question = template.format(name=name)
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                target_ids = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
                target = torch.tensor([target_ids], device='cuda', dtype=torch.long)
                train.append({'name': name, 'value': value, 'prompt': prompt,
                              'slot': slot(value),
                              'ids': torch.cat((prompt['input_ids'], target), dim=1)})

    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [generate(question) for question in ordinary]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4, foreach=False)
    order = list(range(len(train)))
    trace = []
    for step in range(STEPS):
        if step % len(order) == 0:
            random.shuffle(order)
        item = train[order[step % len(order)]]
        prefix = item['prompt']['input_ids'].shape[1]
        target = item['ids'][:, prefix:]
        optimizer.zero_grad(set_to_none=True)
        with use_slot(item['slot']):
            result = model(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                           use_cache=False, return_dict=True)
        scores = result.logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
        loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]),
                                                 target.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite semantic slot loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
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
                answer = generate(question, slot(value))
                rows.append({'group': group, 'name': name, 'value': value,
                             'question': question, 'answer': answer,
                             'no_memory': baseline, 'exact': exact(answer['answer'], value),
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
        summary[group] = {'cases': len(subset), 'exact': sum(row['exact'] for row in subset),
                          'baseline_exact': sum(row['baseline_exact'] for row in subset),
                          'all_values_correct': sum(all(row['exact'] for row in subset
                                                        if row['name'] == name) for name in names),
                          'names': len(names)}

    OUT.mkdir()
    payload = {'format': 'spora-semantic-slot-v1', 'base_sha256': base_sha,
               'rank': RANK, 'steps': STEPS, 'learning_rate': LR, 'seed': SEED,
               'bridge': {name: tensor.detach().cpu() for name, tensor in bridge.state_dict().items()}}
    torch.save(payload, OUT / 'bridge.pt')
    bridge_sha = p.sha(OUT / 'bridge.pt')
    report = {'status': 'semantic_slot_pilot', 'summary': summary, 'rows': rows,
              'trace': trace, 'train_names': crossed.TRAIN_NAMES, 'held_names': crossed.HELD_NAMES,
              'train_values': crossed.TRAIN_VALUES, 'held_values': crossed.HELD_VALUES,
              'base_sha256': base_sha, 'bridge_sha256': bridge_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': [
                  'The writer is oracle-selected: the value token embedding is supplied at write time.',
                  'The slot has no learned natural-language parser, identity router, or correction policy.',
                  'Only one fact relation and short exact answers are evaluated.',
                  'The slot is temporary in this pilot; persistence and independent reload are not tested.',
              ]}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
