"""Read-only routing audit using reviewed OASST user questions, not training."""
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
from scripts.deepseek_memory_intent import IntentGate
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_route_comparison import BINARY
from scripts.deepseek_semantic_gate import hide_numbers
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from spora.finite_question_guard import valid_finite_question_form

REVIEW = ROOT / 'reports/P38-oasst1-chinese-v5-review.json'
SOURCE = ROOT / 'runs/oasst1-chinese-candidate-v5/dialogue.jsonl'
OUTPUT = ROOT / 'runs/deepseek-oasst-route-probe-20261109.json'


def load_questions():
    with REVIEW.open(encoding='utf-8') as stream:
        review = json.load(stream)
    expected = review['manifest']['files']['dialogue.jsonl']['sha256']
    if sha(SOURCE) != expected or review['approved_for_pilot'] is not False:
        raise ValueError('Source or review status changed')
    with SOURCE.open(encoding='utf-8') as stream:
        source = [json.loads(line) for line in stream]
    rows, seen = [], set()
    for item in review['row_reviews']:
        if item['decision'] != 'pass':
            continue
        data = source[item['line'] - 1]
        if data['group'] != item['group'] or data['sha256'] != item['path_sha256']:
            raise ValueError('Review and source row do not match')
        question = data['messages'][0]['content'].strip()
        if data['messages'][0]['role'] != 'user' or question in seen:
            continue
        seen.add(question)
        rows.append({'source_line': item['line'], 'source_group': data['group'], 'question': question})
    return rows


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, BINARY, REVIEW, SOURCE)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    questions = load_questions()
    if not questions:
        raise ValueError('No reviewed examples')
    gate_data = torch.load(BINARY, map_location='cpu', weights_only=True)
    if (gate_data.get('format') != 'deepseek-spora-number-invariant-intent-v1'
            or gate_data.get('weight_sha256') != WEIGHT_SHA or gate_data.get('individual_sha256') != INDIVIDUAL_SHA
            or gate_data.get('layers') != [5, 17] or gate_data.get('threshold') != 0.5):
        raise ValueError('Gate provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    options = [tokenizer.encode(x, add_special_tokens=False) for x in ('是', '否')]
    if any(len(x) != 1 for x in options) or options[0] == options[1]:
        raise ValueError('Expected distinct single-token choices')
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    print('model loaded', flush=True)
    gate = IntentGate(model.config.hidden_size).to('cuda').eval()
    gate.load_state_dict(gate_data['state'], strict=True)
    variants = [(row, name, prefix + row['question']) for row in questions
                for name, prefix in (('original', ''), ('synthetic_prefix', '对象8：'))]
    features, masks = extract(model, tokenizer, [hide_numbers(q) for _, _, q in variants], batch_size=1)
    print('features extracted', flush=True)
    with torch.inference_mode():
        scores = []
        for start in range(0, len(variants), 4):
            scores.extend(gate(features[start:start + 4].to('cuda'),
                               masks[start:start + 4].to('cuda')).sigmoid().cpu().tolist())
    rows = []
    for i, (source, variant, question) in enumerate(variants):
        instruction = ('判断下面的请求是否只要求直接读取单个对象的当前数值。'
                       '历史、来源、否定、写入、计算和包含其他操作的请求都回答否。'
                       '只回答是或否。\n请求：' + question)
        inputs = closed_think_inputs(tokenizer, instruction, 'cuda')
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1, [x[0] for x in options]].float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('Nonfinite score')
            base_score = float(logits.softmax(-1)[0])
        form_ok = valid_finite_question_form(question)
        rows.append(dict(source, variant=variant, tested_question=question, form_ok=form_ok,
                         gate_score=scores[i], base_choice_score=base_score,
                         gate_without_guard=scores[i] >= 0.5,
                         base_without_guard=base_score >= 0.10159165784716606,
                         gate_with_guard=form_ok and scores[i] >= 0.5,
                         base_with_guard=form_ok and base_score >= 0.10159165784716606))
        if (i + 1) % 8 == 0:
            print(f'checked {i + 1}/{len(variants)}', flush=True)
    if any(sha(p) != digest for p, digest in protected.items()):
        raise ValueError('Protected source changed')
    results = {}
    for variant in ('original', 'synthetic_prefix'):
        selected = [r for r in rows if r['variant'] == variant]
        results[variant] = {'examples': len(selected), 'form_ok': sum(r['form_ok'] for r in selected),
                            **{key: sum(r[key] for r in selected) for key in
                               ('gate_without_guard', 'base_without_guard', 'gate_with_guard', 'base_with_guard')}}
    report = {'status': 'reviewed_oasst_negative_development_only', 'source_sha256': protected[SOURCE],
              'review_sha256': protected[REVIEW], 'weight_sha256': WEIGHT_SHA, 'gate_sha256': protected[BINARY],
              'results': results, 'rows': rows, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['OASST source candidate was not approved for training; this is read-only evaluation.',
                              'Previously reviewed pass relates to full dialogue quality, not a routing label.',
                              'Manual audit of included user questions is required before treating them as negative examples.',
                              'Synthetic entity prefix is an artificial stress transformation, not a natural conversation.',
                              'Source is small; the model may have seen similar public data during pretraining.',
                              'Existing base threshold is development-calibrated and score is not confidence.',
                              'No memory reads, writes, answers, or positive recall evaluated.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
