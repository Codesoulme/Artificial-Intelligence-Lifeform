"""Train a gain-based semantic slot bridge on first-answer-token loss."""
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
from spora.research_semantic_slot import install_gain, use_slot


OUT = ROOT / 'runs/deepseek-spora-semantic-slot-gain-pilot-20261116'
RANK = 16
STEPS = 384
LR = 2e-4
INITIAL_GAIN = 32.0
SEED = 84517


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
    bridge = install_gain(model, rank=RANK, initial_gain=INITIAL_GAIN)
    model.requires_grad_(False)
    bridge.requires_grad_(True)
    trainable = list(bridge.parameters())

    def slot(value):
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        return model.model.embed_tokens.weight[token_ids[0]].detach().float()

    train = []
    for name in crossed.TRAIN_NAMES:
        for value in crossed.TRAIN_VALUES:
            for template in crossed.TEMPLATES:
                question = template.format(name=name)
                prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
                target = tokenizer.encode(value, add_special_tokens=False)[0]
                train.append({'name': name, 'value': value, 'prompt': prompt,
                              'slot': slot(value), 'target': target})

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-5, foreach=False)
    order = list(range(len(train)))
    trace = []
    for step in range(STEPS):
        if step % len(order) == 0:
            random.shuffle(order)
        item = train[order[step % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        with use_slot(item['slot']):
            result = model(**item['prompt'], use_cache=False, return_dict=True)
        logits = result.logits[:, -1].float()
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([item['target']], device='cuda'))
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite gain bridge loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({'step': step + 1, 'loss': float(loss.detach()),
                      'gradient_norm': float(gradient),
                      'gain': float(bridge.log_gain.exp().detach())})
        if (step + 1) % 32 == 0:
            print(json.dumps(trace[-1]), flush=True)

    def first_token(name, value):
        question = crossed.TEMPLATES[0].format(name=name)
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        with torch.inference_mode(), use_slot(slot(value)):
            logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1].float()
        target = tokenizer.encode(value, add_special_tokens=False)[0]
        return {'name': name, 'value': value,
                'rank': int((logits > logits[target]).sum()) + 1,
                'winner': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False)}

    rows = [first_token(name, value)
            for name in crossed.HELD_NAMES
            for value in crossed.TRAIN_VALUES + crossed.HELD_VALUES]
    ordinary = ('你好，请简单介绍自己。', '请写一句鼓励学习的话。')
    ordinary_before = [pilot.generate(model, tokenizer, question) for question in ordinary]
    ordinary_after = [pilot.generate(model, tokenizer, question) for question in ordinary]
    if ordinary_before != ordinary_after or p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base or memory-off control changed')
    result = {'status': 'semantic_slot_gain_first_token_pilot',
              'summary': {'seen_value': {'cases': sum(row['value'] in crossed.TRAIN_VALUES for row in rows),
                                         'top1': sum(row['value'] in crossed.TRAIN_VALUES and row['rank'] == 1 for row in rows)},
                          'unseen_value': {'cases': sum(row['value'] in crossed.HELD_VALUES for row in rows),
                                           'top1': sum(row['value'] in crossed.HELD_VALUES and row['rank'] == 1 for row in rows)}},
              'rows': rows, 'trace': trace, 'initial_gain': INITIAL_GAIN,
              'final_gain': float(bridge.log_gain.exp().detach()), 'base_sha256': base_sha,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Value token and write boundary are supplied by the experiment.',
                              'Only first-token decoding is evaluated; no natural writer or persistence.',
                              'One relation and two held-out names only.']}
    OUT.mkdir()
    torch.save({'format': 'spora-semantic-slot-gain-v1', 'base_sha256': base_sha,
                'rank': RANK, 'bridge': {name: tensor.detach().cpu()
                                         for name, tensor in bridge.state_dict().items()}},
               OUT / 'bridge.pt')
    result['bridge_sha256'] = p.sha(OUT / 'bridge.pt')
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
