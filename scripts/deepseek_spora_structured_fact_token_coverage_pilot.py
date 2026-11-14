"""Train a fact-only decoder over a broad fixed token split.

This isolates whether the structured fact lane can support open-ish token
reconstruction. It uses explicit codec packets and does not claim natural
language routing or general conversation.
"""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.structured_text_memory import StructuredTextMemoryCodec
from spora.structured_text_output import StructuredFactSuffixDecoder

OUTPUT = ROOT / 'runs/deepseek-spora-structured-fact-token-coverage-20261115'
TOKEN_SOURCE = ROOT / 'runs/continuous-embedding-bridge-pilot-20261112/result.json'
CHECKPOINT = ROOT / 'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
SEED = 31577
STEPS = 2400
TRAIN_NAMES = ('小林', '小周', '小王')
HELD_NAME = '小赵'


def load_token_split():
    source = json.loads(TOKEN_SOURCE.read_text(encoding='utf-8'))
    ids = source['train_ids'] + source['held_ids']
    if len(ids) != 320 or len(set(ids)) != 320:
        raise ValueError('Unexpected token split')
    return tuple(source['train_ids']), tuple(source['held_ids'])


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = runtime.sha(CHECKPOINT)
    train_ids, held_ids = load_token_split()
    with tempfile.TemporaryDirectory() as temp:
        runtime.CHECKPOINT = CHECKPOINT
        runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = runtime.load_chat(base_loader=lambda: load_full_streamed(skip_replaced=(6, 12)))
        embedding = chat.base.model.embed_tokens.weight.detach()
        tokenizer = chat.tokenizer
        device = embedding.device
        codec = StructuredTextMemoryCodec(
            embedding.cpu(), memory_width=chat.organism.config.memory_width,
            identity_slots=3, fact_slots=5, identity_slot_width=15,
            fact_slot_width=9, seed=29071, eos_token_id=tokenizer.eos_token_id)
        decoder = StructuredFactSuffixDecoder(embedding, codec.config_dict(), hidden=384).to(device)
        cases = []
        for name in TRAIN_NAMES:
            identity_ids = tokenizer.encode(name, add_special_tokens=False)
            for token_id in train_ids:
                packet, metadata = codec.encode(identity_ids, [token_id])
                suffix_ids = tokenizer.encode('喜欢的内容是', add_special_tokens=False) + [token_id]
                suffix_ids += tokenizer.encode('。', add_special_tokens=False) + [tokenizer.eos_token_id]
                cases.append({'group': 'train', 'name': name, 'token_id': token_id,
                              'memory': packet.to(device), 'metadata': metadata,
                              'target': torch.tensor([suffix_ids], dtype=torch.long, device=device)})
        held = []
        identity_ids = tokenizer.encode(HELD_NAME, add_special_tokens=False)
        for token_id in held_ids:
            packet, metadata = codec.encode(identity_ids, [token_id])
            suffix_ids = tokenizer.encode('喜欢的内容是', add_special_tokens=False) + [token_id]
            suffix_ids += tokenizer.encode('。', add_special_tokens=False) + [tokenizer.eos_token_id]
            held.append({'group': 'held', 'name': HELD_NAME, 'token_id': token_id,
                         'memory': packet.to(device), 'metadata': metadata,
                         'target': torch.tensor([suffix_ids], dtype=torch.long, device=device)})
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
                raise ValueError('Nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            if step in (0, 99, 499, 1199, STEPS - 1):
                trace.append({'step': step + 1, 'loss': float(loss.detach()),
                              'grad_norm': float(norm)})

        def evaluate(rows):
            result = []
            for item in rows:
                suffix_ids = decoder.generate(item['memory'].unsqueeze(0),
                                              eos_id=tokenizer.eos_token_id,
                                              max_new_tokens=16)[0].tolist()
                identity_ids = codec.decode(item['memory'].cpu(), item['metadata'])['identity']['token_ids']
                answer_ids = identity_ids + suffix_ids
                result.append({'group': item['group'], 'name': item['name'],
                               'token_id': item['token_id'],
                               'answer': tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                               'suffix_token_ids': suffix_ids,
                               'target_token_in_answer': item['token_id'] in suffix_ids,
                               'first_fact_token': suffix_ids[len(tokenizer.encode('喜欢的内容是', add_special_tokens=False))]
                               if len(suffix_ids) > len(tokenizer.encode('喜欢的内容是', add_special_tokens=False)) else None})
            return result

        train_result, held_result = evaluate(cases), evaluate(held)
    candidate = OUTPUT / 'candidate.pt'
    OUTPUT.mkdir(parents=False, exist_ok=False)
    torch.save({'format': 'spora-structured-fact-token-coverage-v1',
                'state': {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
                'codec': codec.config_dict(), 'source_sha256': source_sha,
                'base_sha256': runtime.WEIGHT_SHA, 'token_source_sha256': runtime.sha(TOKEN_SOURCE),
                'steps': STEPS}, candidate)
    report = {'status': 'structured_fact_token_coverage_pilot',
              'train': train_result, 'held': held_result, 'trace': trace,
              'train_target_in_answer': sum(row['target_token_in_answer'] for row in train_result),
              'held_target_in_answer': sum(row['target_token_in_answer'] for row in held_result),
              'train_cases': len(train_result), 'held_cases': len(held_result),
              'codec': codec.config_dict(), 'candidate_sha256': runtime.sha(candidate),
              'source_sha256': source_sha, 'base_sha256': runtime.WEIGHT_SHA,
              'token_source_sha256': runtime.sha(TOKEN_SOURCE),
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Explicit codec packets and a fixed 320-token split.',
                              'The held split uses a new identity but tokens were selected in advance.',
                              'This tests token reconstruction, not natural language routing or truth.',
                              'The suffix decoder is separate from the DeepSeek conversation path.',
                              'No unknown-state rejection, relation handling, or real individual changed.']}
    (OUTPUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'train_target_in_answer': report['train_target_in_answer'],
                      'held_target_in_answer': report['held_target_in_answer'],
                      'train_cases': report['train_cases'], 'held_cases': report['held_cases']}), flush=True)


if __name__ == '__main__':
    main()
