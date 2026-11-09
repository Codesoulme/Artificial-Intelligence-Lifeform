"""Evaluate thresholds for frozen DeepSeek binary routing on separate examples."""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_guard_adversarial import SECOND_CASES
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_route_comparison import CASES
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form

CALIBRATION = ROOT / 'runs/deepseek-route-comparison-20261109.json'
OUTPUT = ROOT / 'runs/deepseek-route-calibration-20261109.json'
NEW_CASES = (
    (1, 'current', '对象6现在记住的数字是什么？'),
    (1, 'current', '请报出编号22目前所存的数值。'),
    (1, 'current', '告诉我对象35的现有数字。'),
    (1, 'current', '编号51这一刻的值是几？'),
    (1, 'current', '我想获取对象16当前的数据值。'),
    (1, 'current', '查一下编号27现在保存了什么数。'),
    (1, 'current', '对象42的最新记录数字，麻烦给我。'),
    (1, 'current', '请回答编号10的当前值是多少。'),
    (0, 'negated', '别查对象6的当前数字，我不想看。'),
    (0, 'negated', '不用回答编号22的现值了。'),
    (0, 'negated', '撤销查询对象35当前值的请求。'),
    (0, 'negated', '编号51现在的值请暂且保密。'),
    (0, 'source', '对象16当前记录的出处是哪儿？'),
    (0, 'source', '谁告诉你编号27如今的数值？'),
    (0, 'source', '对象42现在的值依据是否可靠？'),
    (0, 'source', '请说明编号10的数值从何而来。'),
    (0, 'history', '对象6刚开始记的数字是什么？'),
    (0, 'history', '编号22更新之前记录了什么？'),
    (0, 'history', '对象35以前都记过什么数？'),
    (0, 'history', '编号51的上一个值是什么？'),
    (0, 'write', '忘掉对象16保存的数字。'),
    (0, 'write', '替换编号27目前的记录。'),
    (0, 'write', '更新对象42的数字并保存。'),
    (0, 'write', '把编号10的记忆删除。'),
    (0, 'derived', '对象6现在的数字是否超过十？'),
    (0, 'derived', '编号22的值减一等于多少？'),
    (0, 'derived', '对象35的现值是偶数吗？'),
    (0, 'derived', '编号51的数值比过去大吗？'),
    (0, 'mixed', '读取对象16现值后清除该记录。'),
    (0, 'mixed', '先删除编号27，然后回答有没有记录。'),
    (0, 'unrelated', '围绕对象42写一句诗。'),
    (0, 'unrelated', '你认为编号10这个名字怎么样？'),
)


def summary(rows, threshold):
    result = {'examples': len(rows), 'false_accept': 0, 'false_defer': 0}
    for row in rows:
        accepted = row['form_ok'] and row['score'] >= threshold
        result['false_accept'] += int(accepted and not row['expected'])
        result['false_defer'] += int(not accepted and row['expected'])
    result['correct'] = len(rows) - result['false_accept'] - result['false_defer']
    return result


def thresholds(rows):
    scores = sorted({row['score'] for row in rows if row['form_ok']})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [1.0]
    positives = sum(r['expected'] for r in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError('Calibration requires both classes')
    def rank(t):
        counts = summary(rows, t)
        return (counts['false_accept'] / negatives + counts['false_defer'] / positives,
                counts['false_accept'], -t)
    balanced = min(candidates, key=rank)
    zero_false = min((t for t in candidates if summary(rows, t)['false_accept'] == 0),
                     key=lambda t: (summary(rows, t)['false_defer'], -t))
    return {'baseline': 0.5, 'balanced': balanced, 'zero_false_calibration': zero_false}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, CALIBRATION)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with CALIBRATION.open(encoding='utf-8') as stream:
        previous = json.load(stream)
    if previous['weight_sha256'] != WEIGHT_SHA or len(previous['rows']) != len(CASES):
        raise ValueError('Unexpected calibration source')
    calibration = []
    for row, (expected, category, question) in zip(previous['rows'], CASES):
        if row['question'] != question or row['expected_current_only'] != expected or row['category'] != category:
            raise ValueError('Calibration labels or questions changed')
        calibration.append({'question': question, 'category': category, 'expected': expected,
                            'form_ok': row['form_ok'], 'score': row['base_yes_choice_probability']})
    cutoffs = thresholds(calibration)
    prior = {hide_numbers(row['question']) for row in calibration}
    held = list(SECOND_CASES) + list(NEW_CASES)
    if prior & {hide_numbers(q) for _, _, q in held}:
        raise ValueError('Calibration and evaluation questions overlap')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    options = [tokenizer.encode(x, add_special_tokens=False) for x in ('是', '否')]
    if any(len(x) != 1 for x in options) or options[0] == options[1]:
        raise ValueError('Expected distinct single-token options')
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    results = {}
    for name, cases in (('previous_stress', SECOND_CASES), ('new_handwritten', NEW_CASES)):
        rows = []
        for expected, category, question in cases:
            instruction = ('判断下面的请求是否只要求直接读取单个对象的当前数值。'
                           '历史、来源、否定、写入、计算和包含其他操作的请求都回答否。'
                           '只回答是或否。\n请求：' + question)
            inputs = closed_think_inputs(tokenizer, instruction, 'cuda')
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[0, -1, [x[0] for x in options]].float()
                score = float(logits.softmax(-1)[0])
            if not math.isfinite(score):
                raise ValueError('Nonfinite score')
            rows.append({'question': question, 'category': category, 'expected': bool(expected),
                         'form_ok': valid_finite_question_form(question), 'score': score})
        results[name] = {'rows': rows, 'thresholds': {k: summary(rows, v) for k, v in cutoffs.items()}}
        print(name + ' complete', flush=True)
    if any(sha(p) != digest for p, digest in protected.items()):
        raise ValueError('Protected source changed')
    report = {'status': 'threshold_development_only', 'weight_sha256': WEIGHT_SHA,
              'calibration_source_sha256': protected[CALIBRATION], 'thresholds': cutoffs,
              'calibration': {k: summary(calibration, v) for k, v in cutoffs.items()}, 'results': results,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All sets are small hand-authored finite examples; previous stress was used in prior gate development.',
                              'Thresholds are fitted to one prompt, one model, and 32 development examples.',
                              'Scores normalize only two forced next-token choices with closed thinking, not confidence.',
                              'No end-to-end memory read, normal chat, writes, or open-domain reliability evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'thresholds': cutoffs, 'calibration': report['calibration'],
                      'results': {k: v['thresholds'] for k, v in results.items()}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
