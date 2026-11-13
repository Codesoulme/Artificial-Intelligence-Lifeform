"""Train a fact-only suffix decoder with codec-authoritative identities."""
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
from spora.structured_text_output import StructuredFactSuffixDecoder


OUTPUT = ROOT / 'runs/deepseek-spora-structured-fact-suffix-decoder-20261113'
CHECKPOINT = ROOT / 'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
SEED = 99132
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
        decoder = StructuredFactSuffixDecoder(embedding, codec.config_dict(), hidden=384).to(device)
        cases = []
        for name in NAMES:
            for word in WORDS:
                packet, metadata = codec.encode(
                    tokenizer.encode(name, add_special_tokens=False),
                    tokenizer.encode(word, add_special_tokens=False))
                suffix = f'喜欢的水果是{word}。'
                target = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
                cases.append({'group': 'train', 'name': name, 'word': word,
                              'memory': packet.to(device), 'metadata': metadata,
                              'target': torch.tensor([target], dtype=torch.long, device=device),
                              'expected_suffix': suffix})
        held = []
        for word in WORDS:
            packet, metadata = codec.encode(
                tokenizer.encode(HELD_NAME, add_special_tokens=False),
                tokenizer.encode(word, add_special_tokens=False))
            suffix = f'喜欢的水果是{word}。'
            target = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
            held.append({'group': 'held_name', 'name': HELD_NAME, 'word': word,
                         'memory': packet.to(device), 'metadata': metadata,
                         'target': torch.tensor([target], dtype=torch.long, device=device),
                         'expected_suffix': suffix})
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
                suffix_ids = decoder.generate(item['memory'].unsqueeze(0),
                                              eos_id=tokenizer.eos_token_id,
                                              max_new_tokens=24)[0].tolist()
                identity_ids = codec.decode(item['memory'].cpu(), item['metadata'])['identity']['token_ids']
                answer = tokenizer.decode(identity_ids + suffix_ids, skip_special_tokens=True).strip()
                expected = f'{item["name"]}{item["expected_suffix"]}'
                result.append({'group': item['group'], 'name': item['name'],
                               'word': item['word'], 'expected': expected,
                               'expected_suffix': item['expected_suffix'],
                               'answer': answer, 'suffix_token_ids': suffix_ids,
                               'identity_token_ids': identity_ids,
                               'exact': answer == expected,
                               'contains_fact': item['word'] in answer})
            return result

        train_result, held_result = evaluate(cases), evaluate(held)
    candidate = OUTPUT / 'candidate.pt'
    OUTPUT.mkdir(parents=False, exist_ok=False)
    torch.save({'format': 'spora-structured-fact-suffix-decoder-v1',
                'state': {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
                'codec': codec.config_dict(), 'source_sha256': source_sha,
                'base_sha256': runtime.WEIGHT_SHA, 'steps': STEPS}, candidate)
    report = {
        'status': 'structured_fact_suffix_decoder_pilot',
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
            'Identity is deterministically decoded by the persisted codec.',
            'The suffix decoder is not general question routing or autonomous writing.',
        ],
    }
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in (
        'train_exact', 'held_exact', 'train_contains_fact', 'held_contains_fact')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
