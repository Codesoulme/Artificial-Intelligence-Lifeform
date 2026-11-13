"""Train a small language-facing adapter on top of TTT-NTP memory reads."""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import (attach_read_adapters, fresh_memory, install,
                                    sequential_ntp_write, use_memory)


OUT = ROOT / 'runs/deepseek-spora-ttt-read-adapter-pilot-20261114'
LAYERS = (6, 12)
RANK = 16
ETA = 0.5
RIDGE = 1.0
STEPS = 96
LR = 3e-3
SEED = 94217
TRAIN = (
    ('小林', '酸奶'), ('阿宁', '豆浆'), ('小雨', '咖啡'), ('阿哲', '果汁'),
    ('小岚', '咖啡'), ('阿明', '酸奶'), ('小周', '豆浆'), ('小吴', '果汁'),
)
HELD = (('小赵', '酸奶'), ('小郑', '豆浆'), ('小王', '咖啡'), ('小孙', '果汁'))


def generate(model, tokenizer, question, memory=None):
    prompt = closed_think_inputs(tokenizer, question, 'cuda')
    kwargs = dict(**prompt, use_cache=True, max_new_tokens=24, do_sample=False,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    with torch.inference_mode():
        if memory is None:
            output = model.generate(**kwargs)
        else:
            with use_memory(memory):
                output = model.generate(**kwargs)
    tail = output[0, prompt['input_ids'].shape[1]:].tolist()
    return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail}


def make_memory(model, tokenizer, wrappers, fact):
    encoded = tokenizer(fact, return_tensors='pt', add_special_tokens=False)
    encoded = {key: value.to('cuda') for key, value in encoded.items()}
    captured = {}
    handles = []
    for layer_id in LAYERS:
        def capture(module, args, output, layer_id=layer_id):
            captured[layer_id] = args[0].detach().float()[0]
        handles.append(wrappers[layer_id].base.register_forward_hook(capture))
    with torch.inference_mode():
        result = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    targets = {}
    keys = {}
    for layer_id in LAYERS:
        layer_hidden = result.hidden_states[layer_id + 1][0].detach().float()
        keys[layer_id] = captured[layer_id][:-1]
        targets[layer_id] = layer_hidden[1:]
    memory = fresh_memory(wrappers)
    sequential_ntp_write(memory, keys, targets, eta=ETA, ridge=RIDGE)
    return {layer_id: value.detach().cpu().half() for layer_id, value in memory.items()}


def device_memory(memory):
    return {layer_id: value.to('cuda', dtype=torch.float32) for layer_id, value in memory.items()}


def build_case(model, tokenizer, wrappers, name, value):
    fact = f'{name}喜欢喝{value}。'
    question = f'请只回答{name}喜欢喝什么饮料。'
    prompt = closed_think_inputs(tokenizer, question, 'cuda')
    target = tokenizer.encode(value, add_special_tokens=False)
    target.append(tokenizer.eos_token_id)
    target_tensor = torch.tensor([target], device='cuda', dtype=torch.long)
    ids = torch.cat((prompt['input_ids'], target_tensor), dim=1)
    return {'name': name, 'value': value, 'fact': fact, 'question': question,
            'prompt': prompt, 'ids': ids,
            'memory': make_memory(model, tokenizer, wrappers, fact)}


def evaluate(model, tokenizer, cases):
    rows = []
    for item in cases:
        memory = device_memory(item['memory'])
        no_memory = generate(model, tokenizer, item['question'])
        with_memory = generate(model, tokenizer, item['question'], memory)
        rows.append({'name': item['name'], 'expected': item['value'], 'question': item['question'],
                     'no_memory': no_memory, 'memory': with_memory,
                     'no_memory_contains': item['value'] in no_memory['answer'],
                     'memory_contains': item['value'] in with_memory['answer'],
                     'tokens_changed': no_memory['token_ids'] != with_memory['token_ids']})
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source_sha = p.sha(p.MODEL / 'model.safetensors')
    if source_sha != p.WEIGHT_SHA:
        raise ValueError('Base model digest mismatch')
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    with tempfile.TemporaryDirectory() as directory:
        model = load_full_streamed().eval()
        wrappers = install(model, LAYERS)
        adapters = attach_read_adapters(wrappers, rank=RANK)
        model.requires_grad_(False)
        trainable = [parameter for adapter in adapters.values() for parameter in adapter.parameters()]
        for parameter in trainable:
            parameter.requires_grad_(True)
        if not trainable:
            raise ValueError('No trainable read adapter parameters')
        train_cases = [build_case(model, tokenizer, wrappers, name, value) for name, value in TRAIN]
        held_cases = [build_case(model, tokenizer, wrappers, name, value) for name, value in HELD]
        ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
        ordinary_before = [generate(model, tokenizer, question) for question in ordinary]
        before = evaluate(model, tokenizer, train_cases + held_cases)
        optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4, foreach=False)
        order = list(range(len(train_cases)))
        trace = []
        for step in range(STEPS):
            if step % len(order) == 0:
                random.shuffle(order)
            item = train_cases[order[step % len(order)]]
            memory = device_memory(item['memory'])
            prefix = item['prompt']['input_ids'].shape[1]
            target = item['ids'][:, prefix:]
            optimizer.zero_grad(set_to_none=True)
            with use_memory(memory):
                result = model(input_ids=item['ids'], attention_mask=torch.ones_like(item['ids']),
                               use_cache=False, return_dict=True)
            scores = result.logits[:, prefix - 1:prefix - 1 + target.shape[1]].float()
            loss = torch.nn.functional.cross_entropy(scores.reshape(-1, scores.shape[-1]), target.reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite adapter loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'gradient_norm': float(gradient)})
            if (step + 1) % 16 == 0:
                print(json.dumps(trace[-1]), flush=True)
        after = evaluate(model, tokenizer, train_cases + held_cases)
        ordinary_after = [generate(model, tokenizer, question) for question in ordinary]
        if ordinary_before != ordinary_after:
            raise ValueError('Memory-off ordinary control changed')
    if p.sha(p.MODEL / 'model.safetensors') != source_sha:
        raise ValueError('Base model changed')
    report = {
        'status': 'ttt_ntp_read_adapter_pilot',
        'before': before, 'after': after,
        'summary': {
            'train_before': sum(row['memory_contains'] for row in before[:len(train_cases)]),
            'train_after': sum(row['memory_contains'] for row in after[:len(train_cases)]),
            'held_before': sum(row['memory_contains'] for row in before[len(train_cases):]),
            'held_after': sum(row['memory_contains'] for row in after[len(train_cases):]),
        },
        'layers': list(LAYERS), 'rank': RANK, 'eta': ETA, 'ridge': RIDGE,
        'steps': STEPS, 'learning_rate': LR, 'trace': trace,
        'base_sha256': source_sha, 'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Eight training and four held-out authored beverage facts only.',
            'Fact ingestion and query templates are program-selected; no natural-language router.',
            'Fast weights are temporary per-case states and are not persisted into the base checkpoint.',
            'The adapter is trained on short exact answers, not open-domain chat.',
            'Held names and values remain within the finite development vocabulary.',
        ],
    }
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
