"""Replay real ordinary replies and isolate the boundary after native reasoning."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_read_adapter_pilot import generate
from scripts.train_spora_recurrent_counterfactual import store_json
from transformers import StoppingCriteria


BUDGET = 72
THINK_BUDGET = 128
SEED = 32639
REMINDER = '请完成最后这条请求，按它指定的语言、数量和格式交付成品。'
HELD = (
    ('translate-confirmation', '把「这次只确认到货时间」翻成英文，只给译文。', (),
     (r'(confirm|verify|check)', r'(delivery|arrival)', r'(only|just)'), (r'怎么|可以|以下',), 90),
    ('translate-alternative', '将「那换成室内活动呢」译成一句英语。', (),
     (r'(what|how) about', r'(indoor|inside)'), (r'outdoor|上一版',), 83),
    ('one-small-action', '你只有一分钟整理一张堆满纸的桌子。只给一个马上可做的动作，一句话。', (),
     (r'纸|文件|桌面', r'收|放|叠|摞|归|整|拿|移|挪|拾'), (r'以下|首先|其次|\d\.\s|\n|还可以',), 34),
    ('shorten-notice', '只保留出门前关窗的提醒，写一句不超过十五字的话。',
     (('user', '出门前检查灯和窗户，提醒我那两件事。'), ('assistant', '出门前关灯，顺手把窗户关好。')),
     (r'关.{0,3}窗|窗.{0,4}关',), (r'灯|检查两',), 15),
    ('update-invitation', '改到周日上午，地点和「带杯子」都保留。给我一句能发给朋友的消息。',
     (('user', '约小孟周六下午到学校东门见，提醒他带杯子。'),
      ('assistant', '小孟，周六下午学校东门见，记得带杯子。')),
     (r'周日.{0,2}上午|星期日.{0,2}上午', r'学校东门', r'带.{0,3}杯'), (r'周六|下午|以下|\n',), 62),
    ('bounded-fact', '快递只显示揽收，页面没给送达时间。现在能不能确定明天到？一句话。', (),
     (r'不.{0,4}确定|无法.{0,4}确定|不能.{0,4}判断',), (r'明天一定|明天肯定|以下',), 43),
    ('plain-mini-answer', '用一句短话告诉我，为什么开着窗帘时室内会更亮？', (),
     (r'光|阳光', r'进|透|照'), (r'氧化|以下|\n',), 42),
)
SMOKE_SCORING = {
    'desk': ((r'桌面|书桌', r'清理|整理'), (r'没有学会|\*\*检查$',), 220),
    'one-step': ((r'桌面|纸|物品', r'清|收|放|整|拿|移|挪|拾'), (r'以下|首先|其次|\d\.\s|\n',), 36),
    'quote-translation': ((r'(?:what|how) about', r'(?:previous|earlier|last) version'),
                          (r'Old Hall|city|Last Book|\n',), 75),
}


class ReasoningEnd(StoppingCriteria):
    def __init__(self, token):
        self.token = token

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[:, -1] == self.token


def cases_from_smoke(payload):
    history, cases = [], []
    for step in payload['first_process']['checks']:
        question, output = step['question'], step['actual']
        kind, answer = output['kind'], output['answer']
        if kind == 'chat':
            name = ('desk', 'one-step', 'quote-translation')[len(cases)]
            cases.append((name, question, tuple(history), *SMOKE_SCORING[name], answer))
        if kind in ('saved', 'reset', 'defer'):
            history.clear()
        elif kind in ('chat', 'read', 'clarify'):
            history = [*history, ('user', question), ('assistant', answer)][-4:]
    if len(cases) != 3:
        raise ValueError('The smoke report did not contain all three ordinary prompts')
    return cases


def assess(answer, ended, required, forbidden, max_chars):
    if isinstance(required, str) or isinstance(forbidden, str):
        raise TypeError('Pass collections of patterns to the reply evaluator')
    checks = {'content': all(re.search(pattern, answer, re.IGNORECASE) for pattern in required),
              'format': len(answer.strip()) <= max_chars and not any(re.search(pattern, answer, re.IGNORECASE)
                                                                     for pattern in forbidden),
              'ended': ended}
    return {**checks, 'pass': all(checks.values())}


def rescore(args, cases):
    stored = json.loads(args.stored.read_text(encoding='utf-8'))
    totals = Counter()
    for case, row in zip(cases, stored['rows'], strict=True):
        name, question, context, required, forbidden, max_chars, _ = case
        if row['name'] != name or row['question'] != question:
            raise ValueError('Stored answer belongs to a different task')
        history = [{'role': role, 'content': text} for role, text in context]
        if 'history' in row and history != row['history']:
            raise ValueError('Stored answer belongs to a different history')
        for mode, output in row['modes'].items():
            output['checks'] = assess(output['answer'], output['eos'], required, forbidden, max_chars)
            totals.update({f'{mode}_pass': output['checks']['pass'], f'{mode}_eos': output['eos']})
    store_json(args.output, {**stored, 'totals': totals,
                             'rescored_from_sha256': base.sha(args.stored), 'scoring_sha256': base.sha(Path(__file__))})
    print(dict(totals), flush=True)


@torch.inference_mode()
def native_pair(model, tokenizer, question, history, seed):
    messages = [*history, {'role': 'user', 'content': question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt.endswith('<think>\n'):
        raise ValueError('Expected the local DeepSeek generation template')
    close, = tokenizer.encode('</think>', add_special_tokens=False)
    separator = tokenizer.encode('\n\n', add_special_tokens=False)
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
    kwargs = {'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': tokenizer.eos_token_id,
              'use_cache': True, 'do_sample': True, 'temperature': .6, 'top_p': .95, 'top_k': 0}
    torch.manual_seed(seed)
    torch.cuda.synchronize()
    started = perf_counter()
    reasoning = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                               max_new_tokens=THINK_BUDGET, stopping_criteria=[ReasoningEnd(close)], **kwargs)
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    count = reasoning.shape[-1] - input_ids.shape[-1]
    last = int(reasoning[0, -1])
    if last == tokenizer.eos_token_id:
        empty = {'answer': '', 'token_ids': [], 'eos': False, 'seconds': 0.}
        return {'ended_before_answer': True, 'reasoning_tokens': count, 'seconds': elapsed}, {
            'close_only': empty, 'close_and_gap': empty}
    if last != close:
        reasoning = torch.cat((reasoning, reasoning.new_tensor([[close]])), dim=-1)
    result = {}
    for mode, suffix in (('close_only', ()), ('close_and_gap', tuple(separator))):
        prefix = torch.cat((reasoning, reasoning.new_tensor([suffix])), dim=-1) if suffix else reasoning
        # Both arms reuse the same sampled prefix and the same continuation seed.
        torch.manual_seed(seed + 1)
        started = perf_counter()
        sequence = model.generate(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                  max_new_tokens=BUDGET, **kwargs)
        torch.cuda.synchronize()
        ids = sequence[0, prefix.shape[-1]:].tolist()
        result[mode] = {'answer': tokenizer.decode(ids, skip_special_tokens=True).strip(), 'token_ids': ids,
                        'eos': bool(ids and ids[-1] == tokenizer.eos_token_id), 'seconds': perf_counter() - started}
    return {'ended_before_answer': False, 'forced': last != close, 'reasoning_tokens': count - int(last == close),
            'seconds': elapsed}, result


def direct(model, tokenizer, question, history):
    torch.cuda.synchronize()
    start = perf_counter()
    result = generate(model, tokenizer, question, history=history, max_new_tokens=BUDGET)
    torch.cuda.synchronize()
    ids = result['token_ids']
    return {**result, 'eos': bool(ids and ids[-1] == tokenizer.eos_token_id), 'seconds': perf_counter() - start}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    smoke = json.loads(args.smoke.read_text(encoding='utf-8'))
    cases = cases_from_smoke(smoke) + [(*row, None) for row in HELD]
    if args.stored:
        rescore(args, cases)
        return
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    files = (args.smoke, Path(__file__), base.MODEL / 'model.safetensors', base.MODEL / 'tokenizer.json',
             base.MODEL / 'tokenizer_config.json')
    sources = {str(path): base.sha(path) for path in files}
    if sources[str(files[2])] != base.WEIGHT_SHA:
        raise ValueError('The DeepSeek weights differ from the local version-memory entry')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    rows, totals = [], Counter()
    for index, (name, question, context, required, forbidden, max_chars, reference) in enumerate(cases):
        history = tuple({'role': role, 'content': text} for role, text in context)
        plain = direct(model, tokenizer, question, history)
        if reference is not None and reference != plain['answer']:
            raise ValueError(f'The actual local reply did not reproduce: {name}')
        guided = direct(model, tokenizer, question + '\n' + REMINDER, history)
        reasoning, pair = native_pair(model, tokenizer, question, history, SEED + index)
        modes = {'direct': plain, 'guided': guided, **pair}
        for mode, output in modes.items():
            output['checks'] = assess(output['answer'], output['eos'], required, forbidden, max_chars)
            totals.update({f'{mode}_pass': output['checks']['pass'], f'{mode}_eos': output['eos']})
        rows.append({'name': name, 'question': question, 'history': history, 'reference': reference,
                     'reasoning': reasoning, 'modes': modes})
        print(name, {mode: result['checks']['pass'] for mode, result in modes.items()},
              'reasoning', reasoning['reasoning_tokens'], flush=True)
    if any(base.sha(path) != sources[str(path)] for path in files):
        raise ValueError('A protected source changed')
    result = {'sources': sources, 'seed': SEED, 'reasoning_budget': THINK_BUDGET, 'visible_budget': BUDGET,
              'rows': rows, 'totals': totals, 'seconds': perf_counter() - started,
              'peak_bytes': torch.cuda.max_memory_allocated()}
    store_json(args.output, result)
    print(dict(totals), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('smoke', type=Path)
    cli.add_argument('output', type=Path)
    cli.add_argument('--stored', type=Path, help='Regrade existing outputs without regenerating text')
    main(cli.parse_args())
