"""Train TTT readers on crossed entity/value assignments and audit swaps."""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import attach_read_adapters, install, use_memory

OUT = ROOT / 'runs/deepseek-spora-ttt-crossed-values-20261114'
TRAIN_NAMES = ('小林', '阿宁', '小雨', '阿哲')
HELD_NAMES = ('小赵', '小郑')
TRAIN_VALUES = ('酸奶', '豆浆', '咖啡', '果汁')
HELD_VALUES = ('牛奶', '红茶')
TEMPLATES = ('请只回答{name}喜欢喝什么饮料。', '记录中，{name}爱喝哪一种饮品？')
ALT_TEMPLATE = '请说出{name}偏爱的饮品名称。'
STEPS = 384
SEED = 59231


def exact(answer, value):
    return answer.strip().rstrip('。.!！ ').strip() == value


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, pilot.LAYERS)
    adapters = attach_read_adapters(wrappers, rank=pilot.RANK)
    model.requires_grad_(False)
    parameters = [param for adapter in adapters.values() for param in adapter.parameters()]
    for param in parameters:
        param.requires_grad_(True)

    memories = {}
    for name in TRAIN_NAMES + HELD_NAMES:
        for value in TRAIN_VALUES + HELD_VALUES:
            if name in TRAIN_NAMES and value in HELD_VALUES:
                continue
            case = pilot.build_case(model, tokenizer, wrappers, name, value, TEMPLATES[0])
            memories[(name, value)] = case['memory']
        print(json.dumps({'prepared': name, 'states': len(memories)}, ensure_ascii=False), flush=True)

    train = []
    for name in TRAIN_NAMES:
        for value in TRAIN_VALUES:
            for template in TEMPLATES:
                question = template.format(name=name)
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                target_ids = tokenizer.encode(value, add_special_tokens=False) + [tokenizer.eos_token_id]
                target = torch.tensor([target_ids], device='cuda', dtype=torch.long)
                train.append({'name': name, 'value': value, 'question': question,
                              'prompt': prompt, 'ids': torch.cat((prompt['input_ids'], target), dim=1)})
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4, foreach=False)
    order = list(range(len(train)))
    trace = []
    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [pilot.generate(model, tokenizer, question) for question in ordinary]
    for step in range(STEPS):
        if step % len(order) == 0:
            rng.shuffle(order)
        item = train[order[step % len(order)]]
        memory = pilot.device_memory(memories[(item['name'], item['value'])])
        prefix = item['prompt']['input_ids'].shape[1]
        target = item['ids'][:, prefix:]
        optimizer.zero_grad(set_to_none=True)
        with use_memory(memory):
            logits = model(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                           use_cache=False, return_dict=True).logits
        scores = logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
        loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 32 == 0:
            row = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(grad)}
            trace.append(row)
            print(json.dumps(row), flush=True)

    rows = []
    for group, names, values, template in (
        ('train_pair', TRAIN_NAMES, TRAIN_VALUES, TEMPLATES[0]),
        ('held_name', HELD_NAMES, TRAIN_VALUES, TEMPLATES[0]),
        ('held_value', HELD_NAMES, HELD_VALUES, TEMPLATES[0]),
        ('held_question', HELD_NAMES, TRAIN_VALUES, ALT_TEMPLATE),
    ):
        for name in names:
            question = template.format(name=name)
            baseline = pilot.generate(model, tokenizer, question)
            for value in values:
                answer = pilot.generate(model, tokenizer, question,
                                        pilot.device_memory(memories[(name, value)]))
                rows.append({'group': group, 'name': name, 'value': value, 'question': question,
                             'answer': answer, 'no_memory': baseline,
                             'exact': exact(answer['answer'], value),
                             'baseline_exact': exact(baseline['answer'], value)})
            print(json.dumps({'evaluated': group, 'name': name}, ensure_ascii=False), flush=True)
    ordinary_after = [pilot.generate(model, tokenizer, question) for question in ordinary]
    if ordinary_before != ordinary_after or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Protected base or memory-off control changed')
    summary = {}
    for group in ('train_pair', 'held_name', 'held_value', 'held_question'):
        subset = [row for row in rows if row['group'] == group]
        names = sorted({row['name'] for row in subset})
        summary[group] = {'cases': len(subset), 'exact': sum(row['exact'] for row in subset),
                          'baseline_exact': sum(row['baseline_exact'] for row in subset),
                          'all_values_correct': sum(all(row['exact'] for row in subset if row['name'] == name)
                                                    for name in names), 'names': len(names)}
    OUT.mkdir()
    payload = {'format': 'spora-ttt-crossed-v1', 'base_sha256': base_sha,
               'layers': list(pilot.LAYERS), 'rank': pilot.RANK, 'seed': SEED,
               'adapters': {str(layer): {name: tensor.detach().cpu() for name, tensor in adapter.state_dict().items()}
                            for layer, adapter in adapters.items()}}
    torch.save(payload, OUT / 'adapters.pt')
    report = {'status': 'crossed_value_development_pilot', 'summary': summary, 'rows': rows,
              'train_names': TRAIN_NAMES, 'held_names': HELD_NAMES, 'train_values': TRAIN_VALUES,
              'held_values': HELD_VALUES, 'train_templates': TEMPLATES, 'steps': STEPS,
              'seed': SEED, 'trace': trace, 'base_sha256': base_sha,
              'adapter_sha256': p.sha(OUT / 'adapters.pt'), 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['One synthetic fact and one explicitly supplied temporary state per query.',
                              'Held values, names, and questions are development probes, not a sealed benchmark.',
                              'Training facts use fixed authored sentence forms and a small finite vocabulary.',
                              'No natural routing, sustained interaction, memory corrections, or durable state test.',
                              'Multiple conditions per name are correlated; exact match can miss valid paraphrases.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
