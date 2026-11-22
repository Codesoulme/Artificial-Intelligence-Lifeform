"""Measure native DeepSeek decoding on the raw model and the anchored SPORA model."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'runs/open-base-deps')]

import torch
from transformers import AutoTokenizer, LogitsProcessor

from scripts.deepseek_local_baseline import MODEL, sha
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_checkpoint import load_integrated_runtime
from spora.language_training import frozen_digest


SCENARIOS = {
    'room_change': (('周四的排练原定在三号教室。只答教室名。', ('三号教室', '3号教室')),
                    ('刚接到通知，改到五号教室。现在去哪间？只答教室名。', ('五号教室', '5号教室')),
                    ('最终还是回原来的教室。只答最新教室名。', ('三号教室', '3号教室'))),
    'inventory': (('木盒里有十三枚棋子，拿走四枚。还剩多少？只答数字。', ('9', '九')),
                  ('我又放进去两枚。现在木盒里有几枚？只答数字。', ('11', '十一')),
                  ('接着拿走三枚。现在还剩多少？只答数字。', ('8', '八'))),
    'negated_facts': (('小蒲不喝橙汁，她喝豆浆。只答她喝的饮品。', ('豆浆',)),
                      ('她为什么不喝橙汁？前面没说原因的话只答不知道。', ('不知道', '不清楚', '未说明')),
                      ('现在她改喝牛奶。只答她现在喝的饮品。', ('牛奶',))),
    'overnight': (('我要给过夜旅行装充电器，给一个最简单的提醒。', ()),
                  ('改成提醒带牙刷，给我一句话。', ()),
                  ('只保留要带的物品。', ('牙刷',))),
}


class ReserveAnswerTokens(LogitsProcessor):
    """Close the reasoning segment if the answer reserve has been reached."""

    def __init__(self, prompt_length, close_at, close_token):
        self.prompt_length = prompt_length
        self.close_at = close_at
        self.close_token = close_token
        self.forced = False

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if self.forced or generated.shape[-1] < self.close_at or (generated == self.close_token).any():
            return scores
        self.forced = True
        scores.fill_(-torch.inf)
        scores[:, self.close_token] = 0
        return scores


@torch.inference_mode()
def run_dialogue(model, tokenizer, turns, *, reasoning, sample, budget, position_limit, seed, reserve=0):
    messages, rows = [], []
    for index, (question, expected) in enumerate(turns):
        messages.append({'role': 'user', 'content': question})
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not rendered.endswith('<think>\n'):
            raise ValueError('Expected the pinned DeepSeek template to open its thinking segment')
        if not reasoning:
            rendered += '</think>\n\n'
        inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt').to('cuda')
        length = inputs['input_ids'].shape[-1]
        actual_budget = min(budget, position_limit - length)
        if actual_budget <= 0:
            raise ValueError('The scenario exhausted the position table')
        policy = []
        if reserve:
            closing = tokenizer.encode('</think>', add_special_tokens=False)
            if len(closing) != 1 or reserve + 1 >= actual_budget:
                raise ValueError('Expected a single delimiter and enough generation budget')
            policy.append(ReserveAnswerTokens(length, actual_budget - reserve - 1, closing[0]))
        torch.manual_seed(seed + index)
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=actual_budget, use_cache=True,
                                   do_sample=sample, temperature=.6 if sample else None,
                                   top_p=.95 if sample else None, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id, logits_processor=policy or None)
        tokens = generated[0, length:].tolist()
        emitted = tokenizer.decode(tokens, skip_special_tokens=True)
        thinking, separator, visible = emitted.partition('</think>') if reasoning else ('', '', emitted)
        answer = visible.strip()
        row = {'question': question, 'answer': answer, 'expected': expected,
               'strict': answer.strip('。！!，,；;？? \n') in expected if expected else None,
               'reasoning_closed': bool(separator) if reasoning else None,
               'reasoning_tokens': len(tokenizer.encode(thinking, add_special_tokens=False)),
               'prompt_tokens': length, 'budget': actual_budget, 'generated_tokens': len(tokens),
               'eos': bool(tokens and tokens[-1] == tokenizer.eos_token_id),
               'seconds': time.perf_counter() - started}
        if policy:
            row['forced_close'] = policy[0].forced
        rows.append(row)
        messages.append({'role': 'assistant', 'content': answer})
    return rows


def evaluate(model, tokenizer, *, budget, limit, label, modes, reserve=0):
    results = {}
    for name, reason, sample in modes:
        results[name] = {}
        for index, (scenario, turns) in enumerate(SCENARIOS.items()):
            rows = run_dialogue(model, tokenizer, turns, reasoning=reason, sample=sample,
                                budget=budget if reason else 48, position_limit=limit, seed=110532 + index * 100,
                                reserve=reserve if reason else 0)
            results[name][scenario] = rows
            print(label, name, scenario, [row['strict'] for row in rows],
                  [row['generated_tokens'] for row in rows], flush=True)
    return results


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    files = (args.checkpoint, MODEL / 'tokenizer_config.json', MODEL / 'generation_config.json')
    sources = {str(path): sha(path) for path in files}
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    parent = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    limit = parent['metadata']['position_table_max_position']
    modes = [('closed_greedy', False, False), ('closed_sampled', False, True), ('native_sampled', True, True)]
    if args.native_only:
        modes = modes[-1:]
    results = {}
    if not args.spora_only:
        results['original'] = evaluate(base, tokenizer, budget=args.budget, limit=limit, label='original',
                                       modes=modes, reserve=args.reserve_answer)
    runtime = load_integrated_runtime(args.checkpoint, base, MODEL)
    runtime.decoder.eval().requires_grad_(False)
    individual = frozen_digest(runtime.organism, runtime.state)
    results['spora'] = evaluate(runtime.decoder, tokenizer, budget=args.budget, limit=limit, label='spora',
                                modes=modes, reserve=args.reserve_answer)
    if individual != frozen_digest(runtime.organism, runtime.state) or any(
        sha(Path(path)) != digest for path, digest in sources.items()
    ):
        raise ValueError('The reasoning audit changed a protected input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'results': results, 'budget': args.budget, 'position_limit': limit,
              'reserve_answer': args.reserve_answer,
              'layout': runtime.decoder.anchored_attention_layout, 'sources': sources,
              'seconds': time.perf_counter() - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--native-only', action='store_true')
    parser.add_argument('--spora-only', action='store_true')
    parser.add_argument('--reserve-answer', type=int, default=0)
    args = parser.parse_args()
    if args.budget < 1:
        parser.error('--budget must be positive')
    if not 0 <= args.reserve_answer < args.budget or (args.reserve_answer and not args.native_only):
        parser.error('--reserve-answer requires --native-only and a reserve below the generation budget')
    main(args)
