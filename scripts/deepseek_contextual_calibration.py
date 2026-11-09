"""Test content-free choice-bias correction on recorded routing examples."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA
from scripts.deepseek_route_calibration import CALIBRATION, OUTPUT as HELD, summary
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha

OUTPUT = ROOT / 'runs/deepseek-contextual-calibration-20261109.json'
CONTENT_FREE = ('N/A', '无', '')


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    protected = {p: sha(p) for p in (MODEL / 'model.safetensors', INDIVIDUAL, CALIBRATION, HELD)}
    if protected[MODEL / 'model.safetensors'] != WEIGHT_SHA or protected[INDIVIDUAL] != INDIVIDUAL_SHA:
        raise ValueError('Protected model mismatch')
    with CALIBRATION.open(encoding='utf-8') as stream:
        calibration = json.load(stream)
    with HELD.open(encoding='utf-8') as stream:
        held = json.load(stream)
    if (calibration['weight_sha256'] != WEIGHT_SHA or held['weight_sha256'] != WEIGHT_SHA
            or held['calibration_source_sha256'] != protected[CALIBRATION]):
        raise ValueError('Source provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    options = [tokenizer.encode(x, add_special_tokens=False) for x in ('是', '否')]
    if any(len(x) != 1 for x in options) or options[0] == options[1]:
        raise ValueError('Expected distinct single-token options')
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    priors = {}
    for content in CONTENT_FREE:
        prompt = ('判断下面的请求是否只要求直接读取单个对象的当前数值。'
                  '历史、来源、否定、写入、计算和包含其他操作的请求都回答否。'
                  '只回答是或否。\n请求：' + content)
        inputs = closed_think_inputs(tokenizer, prompt, 'cuda')
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[0, -1, [x[0] for x in options]].float()
            score = float(logits.softmax(-1)[0])
        if not math.isfinite(score) or not 0 < score < 1:
            raise ValueError('Invalid content-free score')
        priors[content or '<empty>'] = score
    if any(sha(p) != digest for p, digest in protected.items()):
        raise ValueError('Protected source changed')
    # In a binary decision, dividing each choice probability by its content-free
    # probability is equivalent to thresholding the uncorrected score at the prior.
    datasets = {'calibration': [{'question': r['question'], 'category': r['category'],
                                 'expected': r['expected_current_only'], 'form_ok': r['form_ok'],
                                 'score': r['base_yes_choice_probability']} for r in calibration['rows']]}
    datasets.update({name: value['rows'] for name, value in held['results'].items()})
    results = {name: {prior: summary(rows, threshold) for prior, threshold in priors.items()}
               for name, rows in datasets.items()}
    report = {'status': 'content_free_choice_bias_development_only', 'weight_sha256': WEIGHT_SHA,
              'calibration_source_sha256': protected[CALIBRATION], 'held_source_sha256': protected[HELD],
              'content_free_priors': priors, 'results': results,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three content-free strings may themselves have task-specific interpretations.',
                              'Same prompt, closed-thinking prefix and two choice tokens as previous experiments.',
                              'All sets are small development examples and share one model and task.',
                              'Not calibrated confidence, not a test of open routing or end-to-end memory.',
                              'No selection between reference strings should be evaluated on the same data.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'priors': priors, 'results': results}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
