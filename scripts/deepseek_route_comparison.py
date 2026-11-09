"""Compare frozen-model routing and learned gates on fresh finite examples."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_guard_adversarial import CASES as OLD_CASES, SECOND_CASES
from scripts.deepseek_memory_intent import IntentGate, build_cases
from scripts.deepseek_multitype_gate import CLASSES, THRESHOLD, MultiIntentGate, infer
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form

OUTPUT = ROOT / 'runs/deepseek-route-comparison-20261109.json'
BINARY = ROOT / 'runs/deepseek-semantic-gate-2000-20261109/gate.pt'
MULTI = ROOT / 'runs/deepseek-multitype-gate-2000-20261109/gate.pt'

# These labels describe eligibility for the single-current-value read path only.
CASES = (
    (True, 'current', '麻烦把对象3现在保存的那个数字告诉我。'),
    (True, 'current', '我需要编号21当下的记录值。'),
    (True, 'current', '对象45眼下对应的数值具体为多少？'),
    (True, 'current', '能查查编号2当前存的是哪个数吗？'),
    (True, 'current', '对象31此时的数值，请直接报出来。'),
    (True, 'current', '请给出编号54最新保存的数字。'),
    (True, 'current', '对象18当前记着什么数？'),
    (True, 'current', '对于编号63，当前记录的数值是什么？'),
    (False, 'negated', '先别告诉我对象3现在保存的数字。'),
    (False, 'negated', '我不需要编号21当前的数值，请跳过。'),
    (False, 'negated', '停止查询对象45的值，我改主意了。'),
    (False, 'negated', '请忽略编号2目前是多少这个问题。'),
    (False, 'source', '谁最早向你提供对象31的当前记录？'),
    (False, 'source', '编号54的当前数值可信度如何？'),
    (False, 'source', '你是通过什么途径知道对象18的现值的？'),
    (False, 'source', '编号63当前数字的证据是什么？'),
    (False, 'history', '对象3变成现在这样之前记录了什么数？'),
    (False, 'history', '回忆一下编号21第一次记下的数字。'),
    (False, 'history', '对象45曾经保存的数值有哪些？'),
    (False, 'history', '编号2上次被修改前是什么数字？'),
    (False, 'write', '请抹去对象31现有记录。'),
    (False, 'write', '把编号54的值改掉，不用先报给我。'),
    (False, 'write', '更新对象18的记忆，替换目前的数据。'),
    (False, 'write', '把编号63标记为没有记录。'),
    (False, 'derived', '对象3的当前数字能被二整除吗？'),
    (False, 'derived', '编号21的当前值与之前相比增加了吗？'),
    (False, 'derived', '如果对象45的数值乘以二会得到什么？'),
    (False, 'derived', '判断编号2现在的值是否大于十。'),
    (False, 'mixed', '读出对象31的当前值，然后把它删掉。'),
    (False, 'mixed', '先更新编号54的记录，再告诉我结果。'),
    (False, 'unrelated', '帮我编一个关于对象18的小故事。'),
    (False, 'unrelated', '你觉得编号63这个称呼好听吗？'),
)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, BINARY, MULTI)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    binary_ckpt = torch.load(BINARY, map_location='cpu', weights_only=True)
    multi_ckpt = torch.load(MULTI, map_location='cpu', weights_only=True)
    for data in (binary_ckpt, multi_ckpt):
        if data.get('weight_sha256') != WEIGHT_SHA or data.get('individual_sha256') != INDIVIDUAL_SHA:
            raise ValueError('Gate provenance mismatch')
    if (multi_ckpt.get('layers') != [5, 17] or tuple(multi_ckpt['classes']) != CLASSES
            or multi_ckpt.get('threshold') != THRESHOLD):
        raise ValueError('Unexpected multi-type gate')
    if (binary_ckpt.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or binary_ckpt.get('layers') != [5, 17] or binary_ckpt.get('threshold') != 0.5):
        raise ValueError('Unexpected binary gate')
    old = {hide_numbers(row['question']) for held in (False, True)
           for row in build_cases(held, augmented=True, malformed=True, contrastive=True)}
    old.update(hide_numbers(row[2]) for row in OLD_CASES + SECOND_CASES)
    if old.intersection(hide_numbers(row[2]) for row in CASES):
        raise ValueError('New questions overlap previous examples after number hiding')
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    options = [tokenizer.encode(x, add_special_tokens=False) for x in ('是', '否')]
    if any(len(x) != 1 for x in options) or options[0] == options[1]:
        raise ValueError('Expected distinct single-token options')
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    binary = IntentGate(model.config.hidden_size).to('cuda').eval()
    binary.load_state_dict(binary_ckpt['state'], strict=True)
    multi = MultiIntentGate(model.config.hidden_size).to('cuda').eval()
    multi.load_state_dict(multi_ckpt['state'], strict=True)
    features, masks = extract(model, tokenizer, [hide_numbers(q) for _, _, q in CASES])
    with torch.inference_mode():
        binary_scores = binary(features.to('cuda'), masks.to('cuda')).sigmoid().cpu().tolist()
    multi_scores = infer(multi, features, masks)
    rows = []
    for i, (expected, category, question) in enumerate(CASES):
        instruction = ('判断下面的请求是否只要求直接读取单个对象的当前数值。'
                       '历史、来源、否定、写入、计算和包含其他操作的请求都回答否。'
                       '只回答是或否。\n请求：' + question)
        inputs = closed_think_inputs(tokenizer, instruction, 'cuda')
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1, [x[0] for x in options]].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite model logits')
            yes_score = float(logits.softmax(-1)[0])
        form_ok = valid_finite_question_form(question)
        predicted = CLASSES[max(range(len(CLASSES)), key=lambda j: multi_scores[i][j])]
        rows.append({'question': question, 'category': category, 'expected_current_only': expected,
                     'form_ok': form_ok,
                     'binary_probability': binary_scores[i], 'binary_accept': form_ok and binary_scores[i] >= 0.5,
                     'multi_probability': multi_scores[i][0], 'multi_type': predicted,
                     'multi_accept': form_ok and predicted == 'current' and multi_scores[i][0] >= THRESHOLD,
                     'base_yes_choice_probability': yes_score, 'base_accept': form_ok and yes_score >= 0.5})
        if (i + 1) % 8 == 0:
            print(f'checked {i + 1}/{len(CASES)}', flush=True)
    if any(sha(path) != digest for path, digest in protected.items()):
        raise ValueError('Protected source changed')
    summary = {}
    for method in ('binary', 'multi', 'base'):
        summary[method] = {'correct': sum(row[method + '_accept'] == row['expected_current_only'] for row in rows),
                           'false_accept': sum(row[method + '_accept'] and not row['expected_current_only'] for row in rows),
                           'false_defer': sum(not row[method + '_accept'] and row['expected_current_only'] for row in rows)}
    report = {'status': 'handwritten_development_comparison_only', 'rows': rows, 'summary': summary,
              'weight_sha256': WEIGHT_SHA, 'gate_sha256': {p.name + '_' + p.parent.name: protected[p] for p in (BINARY, MULTI)},
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Small hand-authored set; labels and prompt are task-specific, not independent broad evaluation.',
                              'Base compares the next-token logits of two forced options with a closed thinking prefix.',
                              'Choice probabilities are conditional on two candidate tokens, not calibrated confidence.',
                              'All paths use the finite symbolic format guard; no memory read or generation is tested.',
                              'No write operations or changes to protected model/state.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
