"""Isolated full-sentence training with counterfactual native memory values."""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta

OUT = ROOT / 'runs/deepseek-spora-sentence-lora-pilot-20261114'
LAYERS = (18, 23)
RANK = 8
STEPS = 64
SEED = 73119
TRAIN_TEMPLATES = ('请用完整的一句话回答{name}喜欢喝什么饮料。',
                   '用一句话说明记录中{name}的饮品偏好。')
HELD_TEMPLATE = '根据已有记忆，写一句话告诉我{name}偏爱哪种饮品。'
GENERIC = ('请写一句鼓励学习的话。', '一加一等于多少？')


def make_state(chat, key, value):
    state = p.fresh_controlled_state(chat.organism)
    state, decision = p.observe_sourced(chat.organism, state, key, value, 'direct', 1.0)
    if not decision['current_write']:
        raise ValueError('Controlled observation rejected')
    return state


def answer(chat, question, args=None):
    prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
    with torch.inference_mode():
        ids = chat.decoder.generate(**prefix, **({} if args is None else args),
                                    max_new_tokens=32, do_sample=False,
                                    pad_token_id=chat.tokenizer.pad_token_id,
                                    eos_token_id=chat.tokenizer.eos_token_id)
    tail = ids[0, prefix['input_ids'].shape[1]:].tolist()
    return {'text': chat.tokenizer.decode(tail, skip_special_tokens=True).strip(), 'token_ids': tail}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    rng = random.Random(SEED)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-sentence-lora-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        chat.base.requires_grad_(False)
        chat.decoder.reader.requires_grad_(False)
        modules = []
        for layer in LAYERS:
            mlp = chat.base.model.layers[layer].mlp
            module = OnlineDelta(mlp.down_proj, RANK)
            mlp.down_proj = module
            modules.append(module)
        parameters = [param for module in modules for projection in (module.down, module.up)
                      for param in projection.parameters()]
        train_rows = [(index, value, template) for index in (0, 1) for value in (1, 2, 3, 6)
                      for template in TRAIN_TEMPLATES]
        held = [(index, value) for index in (6, 7) for value in (4, 7)]
        states = {(index, value): make_state(chat, KEYS[index], value)
                  for index, value in {(i, v) for i, v, _ in train_rows} | set(held)}
        state_digests = {key: p.frozen_digest(chat.organism, state) for key, state in states.items()}

        def evaluate():
            rows = []
            for index, value in held:
                state = states[index, value]
                args, logits = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
                question = HELD_TEMPLATE.format(name=NAMES[index])
                with_memory = answer(chat, question, args)
                without_memory = answer(chat, question)
                rows.append({'name': NAMES[index], 'key': KEYS[index], 'value': p.WORDS[value],
                             'native_correct': int(logits.argmax(-1)[0]) == value, 'question': question,
                             'with_memory': with_memory, 'without_memory': without_memory})
            return {'rows': rows, 'generic': [answer(chat, question) for question in GENERIC]}

        before = evaluate()
        optimizer = torch.optim.AdamW(parameters, lr=0.0005, weight_decay=0, foreach=False)
        trace, order = [], []
        for step in range(STEPS):
            if not order:
                order = list(train_rows)
                rng.shuffle(order)
            index, value, template = order.pop()
            question = template.format(name=NAMES[index])
            target_text = f'{NAMES[index]}喜欢喝{p.WORDS[value]}。'
            prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
            target = chat.tokenizer(target_text, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
            target = torch.cat((target, torch.tensor([[chat.tokenizer.eos_token_id]], device='cuda')), dim=1)
            inputs = torch.cat((prefix['input_ids'], target), dim=1)
            attention_mask = torch.ones_like(inputs)
            args, _ = p.runtime._prepare_memory_read(chat.organism, states[index, value], KEYS[index], 'current', None)
            optimizer.zero_grad(set_to_none=True)
            output = chat.decoder(input_ids=inputs, attention_mask=attention_mask, **args)
            start = prefix['input_ids'].shape[1] - 1
            logits = output.logits[:, start:start + target.shape[1]].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 7, 15, 31, 47, 63):
                point = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(grad)}
                trace.append(point)
                print(json.dumps(point), flush=True)
        after = evaluate()
        for module in modules:
            module.enabled = False
        disabled = evaluate()
        if disabled != before:
            raise ValueError('Disabled adapter does not reproduce original output')
        if p.frozen_digest(chat.organism, chat.state) != original_digest or any(
                p.frozen_digest(chat.organism, state) != state_digests[key] for key, state in states.items()):
            raise ValueError('Memory state changed')
        candidate = [{key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                      if key.startswith(('down.', 'up.'))} for module in modules]
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected files changed')
    OUT.mkdir()
    model_path = OUT / 'adapter.pt'
    torch.save({'format': 'spora-sentence-lora-pilot-v1', 'base_sha256': base_sha,
                'checkpoint_sha256': checkpoint_sha, 'layers': LAYERS, 'rank': RANK,
                'adapters': candidate}, model_path)
    result = {'before': before, 'after': after, 'disabled_matches_before': True,
              'train_rows': len(train_rows), 'steps': STEPS, 'seed': SEED, 'loss_trace': trace,
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha, 'adapter_sha256': p.sha(model_path),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Synthetic finite-value data and manually supplied numeric addresses.',
                              'Held names and templates have been used elsewhere; values omitted only from this training.',
                              'Fixed target sentence format can cause template overfitting.',
                              'No natural write, router integration, persistent individual or independent evaluation.',
                              'Only two generic prompts and no formal base quality preservation guarantee.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'after': after, 'seconds': result['seconds']}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
