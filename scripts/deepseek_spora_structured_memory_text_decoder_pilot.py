"""Train a structured-lane memory-to-text decoder on an isolated candidate."""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch

from scripts import deepseek_spora_runtime_chat as runtime
from spora.structured_text_memory import StructuredTextMemoryCodec
from spora.structured_text_output import StructuredMemoryTextDecoder


OUTPUT = ROOT / 'runs/deepseek-spora-structured-memory-text-decoder-20261113'
CHECKPOINT = ROOT / 'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
SEED = 99131
STEPS = 1000
NAMES = ('小林', '小周', '小王')
HELD_NAME = '小赵'
WORDS = ('苹果', '香蕉', '西瓜', '葡萄')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    started = time.perf_counter()
    source_sha = runtime.sha(CHECKPOINT)
    with tempfile.TemporaryDirectory() as temp:
        runtime.CHECKPOINT = CHECKPOINT
        runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = runtime.load_chat()
        embedding = chat.base.model.embed_tokens.weight.detach()
        tokenizer = chat.tokenizer
        device = embedding.device
        codec = StructuredTextMemoryCodec(
            embedding.cpu(), memory_width=chat.organism.config.memory_width,
            identity_slots=3, fact_slots=5, identity_slot_width=15,
            fact_slot_width=9, seed=29071, eos_token_id=tokenizer.eos_token_id)
        decoder = StructuredMemoryTextDecoder(
            embedding, memory_width=codec.memory_width, slot_count=codec.total_slots,
            lane_specs=((codec.identity_slots, codec.identity_slot_width),
                        (codec.fact_slots, codec.fact_slot_width)), hidden=384).to(device)
        cases = []
        for name in NAMES:
            for word in WORDS:
                packet, _ = codec.encode(
                    tokenizer.encode(name, add_special_tokens=False),
                    tokenizer.encode(word, add_special_tokens=False))
                answer = f'{name}喜欢的水果是{word}。'
                target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                cases.append({'group': 'train', 'name': name, 'word': word,
                              'memory': packet.to(device),
                              'target': torch.tensor([target], dtype=torch.long, device=device),
                              'expected': answer})
        held = []
        for word in WORDS:
            packet, _ = codec.encode(
                tokenizer.encode(HELD_NAME, add_special_tokens=False),
                tokenizer.encode(word, add_special_tokens=False))
            answer = f'{HELD_NAME}喜欢的水果是{word}。'
            target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            held.append({'group': 'held_name', 'name': HELD_NAME, 'word': word,
                         'memory': packet.to(device),
                         'target': torch.tensor([target], dtype=torch.long, device=device),
                         'expected': answer})
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=2e-3,
                                      weight_decay=1e-4, foreach=False)
        order = list(range(len(cases)))
        trace = []
        for step in range(STEPS):
            if step % len(order) == 0:
                random.shuffle(order)
            item = cases[order[step % len(order)]]
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(item['memory'].unsqueeze(0), item['target']).float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), item['target'].reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite decoder loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 49, 199, 499, STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach()),
                              'grad_norm': float(norm)})

        def evaluate(rows):
            result = []
            for item in rows:
                token_ids = decoder.generate(item['memory'].unsqueeze(0),
                                             eos_id=tokenizer.eos_token_id,
                                             max_new_tokens=32)[0].tolist()
                answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                result.append({'group': item['group'], 'name': item['name'],
                               'word': item['word'], 'expected': item['expected'],
                               'answer': answer, 'token_ids': token_ids,
                               'exact': answer == item['expected'],
                               'contains_fact': item['word'] in answer})
            return result

        train_result, held_result = evaluate(cases), evaluate(held)
    candidate = OUTPUT / 'candidate.pt'
    OUTPUT.mkdir(parents=False, exist_ok=False)
    torch.save({'format': 'spora-structured-memory-text-decoder-v1',
                'state': {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
                'codec': codec.config_dict(), 'source_sha256': source_sha,
                'base_sha256': runtime.WEIGHT_SHA, 'steps': STEPS}, candidate)
    report = {
        'status': 'structured_memory_text_decoder_pilot',
        'train': train_result, 'held': held_result,
        'train_exact': sum(row['exact'] for row in train_result),
        'held_exact': sum(row['exact'] for row in held_result),
        'train_contains_fact': sum(row['contains_fact'] for row in train_result),
        'held_contains_fact': sum(row['contains_fact'] for row in held_result),
        'trace': trace, 'codec': codec.config_dict(),
        'candidate_sha256': runtime.sha(candidate), 'source_sha256': source_sha,
        'base_sha256': runtime.WEIGHT_SHA, 'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Explicit structured packets and a finite authored fruit vocabulary.',
            'The held person is unseen during decoder training but uses the same lane codec.',
            'The decoder is not yet connected to general question routing or automatic writes.',
        ],
    }
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'train_exact', 'held_exact', 'train_contains_fact', 'held_contains_fact')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
