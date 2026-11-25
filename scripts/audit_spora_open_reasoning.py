"""Compare bounded native reasoning and direct replies on frozen Chinese histories."""

import argparse
from collections import Counter
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from scripts.audit_spora_open_chinese import BUDGET, compare_deepseek, response_checks, store_json
from scripts.audit_spora_reasoning_budget import ReserveAnswerTokens
from scripts.deepseek_local_baseline import sha, torch
from transformers import StoppingCriteria


REASONING_BUDGET = 215
POSITION_LIMIT = 768
SEED = 11255


class VisibleReplyLimit(StoppingCriteria):
    def __init__(self, prompt_length, delimiter, budget):
        self.prompt_length = prompt_length
        self.delimiter = delimiter
        self.budget = budget
        self.end = None
        self.answer_started = None

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[-1] - self.prompt_length
        if self.end is None and int(input_ids[0, -1]) == self.delimiter:
            self.end = generated
            self.answer_started = perf_counter()
        return self.end is not None and generated - self.end >= self.budget


@torch.inference_mode()
def run_reply(model, tokenizer, messages, *, reasoning, seed):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not text.endswith('<think>\n'):
        raise ValueError('Expected the pinned DeepSeek chat template')
    if not reasoning:
        text += '</think>\n\n'
    inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
    length = inputs.input_ids.shape[-1]
    budget = BUDGET + (REASONING_BUDGET + 1 if reasoning else 0)
    if length + budget > POSITION_LIMIT:
        raise ValueError('The frozen prompt exceeded the position budget')
    reserve, limit = None, None
    if reasoning:
        close, = tokenizer.encode('</think>', add_special_tokens=False)
        reserve = ReserveAnswerTokens(length, REASONING_BUDGET, close)
        limit = VisibleReplyLimit(length, close, BUDGET)
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    started = perf_counter()
    sequence = model.generate(**inputs, use_cache=True, do_sample=True, temperature=.6, top_p=.95, top_k=0,
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                              max_new_tokens=budget, logits_processor=[reserve] if reserve else None,
                              stopping_criteria=[limit] if limit else None)
    torch.cuda.synchronize()
    ended = perf_counter()
    tokens = sequence[0, length:].tolist()
    eos = bool(tokens and tokens[-1] == tokenizer.eos_token_id)
    boundary = limit.end if limit else 0
    if limit and boundary is not None and tokens[boundary - 1] != close:
        raise ValueError('Thinking delimiter did not match the generated token sequence')
    visible = tokens[boundary:] if boundary is not None else []
    preview = tokenizer.decode(visible, skip_special_tokens=True).strip()
    answer_started = (limit.answer_started or ended) if limit else started
    return {'answer': preview, 'prompt_tokens': length, 'generated_tokens': len(tokens),
            'reasoning_tokens': len(tokens) - int(eos) if boundary is None else max(0, boundary - 1),
            'visible_tokens': len(visible) - int(eos and bool(visible)),
            'reasoning_closed': boundary is not None if reasoning else None,
            'forced_close': bool(reserve and reserve.forced), 'eos': eos, 'seed': seed,
            'seconds': ended - started, 'reasoning_seconds': answer_started - started,
            'visible_seconds': ended - answer_started}


def contexts(case):
    turns = case['turns']
    if not case['resolved_prompt']:
        return {'ordinary': [{'role': 'user', 'content': turns[0]}]}
    messages = []
    for prompt, row in zip(turns[:-1], case['rollout'][:-1], strict=True):
        messages.extend(({'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': row['answer']}))
    messages.append({'role': 'user', 'content': turns[-1]})
    return {'raw': messages, 'resolved': [{'role': 'user', 'content': case['resolved_prompt']}]}


def evaluate(previous, output, baseline_sha, label, model, tokenizer):
    rows, counts = [], Counter()
    for index, case in enumerate(previous[label]['cases']):
        for mode, messages in contexts(case).items():
            pair = {'direct_greedy': case['conditions'][mode]}
            for name, reasoning in (('direct_sampled', False), ('native_sampled', True)):
                result = run_reply(model, tokenizer, messages, reasoning=reasoning, seed=SEED + index)
                result['checks'] = response_checks(result['answer'], case['required'], case['forbidden'])
                pair[name] = result
            for name, result in pair.items():
                key = f'{mode}_{name}'
                counts.update({f'{key}_pass': result['checks']['surface_pass'], f'{key}_eos': result['eos'],
                               f'{key}_forced': result.get('forced_close', False)})
            print(label, case['name'], mode, {name: (row['checks']['surface_pass'], row.get('reasoning_tokens', 0))
                                              for name, row in pair.items()}, flush=True)
            rows.append({'name': case['name'], 'mode': mode, 'messages': messages, 'pair': pair})
    result = {'rows': rows, 'summary': dict(counts), 'baseline_sha256': baseline_sha}
    store_json(output / f'{label}.json', result)
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f'{name}.json').exists() for name in ('native', 'integrated', 'result')):
        raise FileExistsError(args.output)
    baseline_sha = sha(args.baseline)
    previous = json.loads(args.baseline.read_text(encoding='utf-8'))
    manifest = Path(__file__).with_name('spora_open_chinese_cases.py')
    if previous['cases_sha256'] != sha(manifest) or previous['budget'] != BUDGET:
        raise ValueError('Expected the previous comparison and case manifest')
    result = compare_deepseek(args.checkpoint, partial(evaluate, previous, args.output, baseline_sha))
    if (sha(args.baseline) != baseline_sha or sha(manifest) != previous['cases_sha256']
            or any(result[key] != previous[key] for key in ('sources', 'individual_sha256'))):
        raise ValueError('The reasoning study changed a protected source or used a different individual')
    store_json(args.output / 'result.json', {**result, 'baseline_sha256': baseline_sha,
                                             'visible_budget': BUDGET, 'reasoning_budget': REASONING_BUDGET,
                                             'position_limit': POSITION_LIMIT, 'cases_sha256': sha(manifest)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('runs/named-memory-layer9-packed-reply-v1/packed.pt'))
    main(parser.parse_args())
