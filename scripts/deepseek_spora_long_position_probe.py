"""Probe zero-initialized position extension without changing old weights."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'runs/open-base-deps'))

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_integrated_cache_dialogue_probe import POST

OUTPUT = ROOT/'runs/deepseek-spora-long-position-probe-20261111.json'
CASES = (
    {'name': 'schedule', 'prompt': (
        '我正在和朋友规划一次周末出行。我们先讨论过天气，也看过公共交通路线。'
        '最初打算星期六上午去城市公园，后来发现那天有一场持续一整天的雨，'
        '大家就决定更改计划。小林建议去室内美术馆，小周表示她需要下午四点以前回家。'
        '我们还商量了午餐，最后决定在美术馆附近的一家面馆吃饭。'
        '接着我查了美术馆的开放时间，星期天下午可以参观。'
        '大家最后约定星期天下午一点半在美术馆门口集合，参观结束后各自回家。'
        '请只回答：我们最终约定在哪里集合？'), 'reference': '美术馆门口'},
    {'name': 'correction', 'prompt': (
        '我正在整理一份家庭采购清单，请仔细区分原计划和后来的变更。'
        '今天早上我打算买牛奶、苹果、面条和一包纸巾。'
        '下午家人告诉我冰箱里还有足够的牛奶，所以不用再买牛奶。'
        '家里的苹果也刚买过，于是把苹果从清单里划掉。'
        '面条已经吃完，需要保留在清单上。纸巾数量不多，也仍然需要买。'
        '后来家人提出还需要一瓶酱油，我把酱油添加到了清单的末尾。'
        '购物前我再次核对：面条、纸巾、酱油需要买，牛奶和苹果不需要买。'
        '请只回答：后来新添加到购物清单的是什么？'), 'reference': '酱油'},
)


def answer_metrics(base, tokenizer, prefix, reference):
    answer = tokenizer(reference, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
    ids = torch.cat((prefix['input_ids'], answer), dim=1)
    with torch.inference_mode():
        logits = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
        start = prefix['input_ids'].shape[1]-1
        selected = logits[:, start:start+answer.shape[1]]
        nll = F.cross_entropy(selected.reshape(-1, selected.shape[-1]), answer.reshape(-1))
        generated = base.generate(**prefix, use_cache=True, max_new_tokens=24, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tail = generated[0, prefix['input_ids'].shape[1]:]
    return {'reference_mean_nll': float(nll), 'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'generated_tokens': tail.numel(), 'ended_with_eos': bool(tail.numel() and tail[-1] == tokenizer.eos_token_id)}


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if sha(MODEL/'model.safetensors') != WEIGHT_SHA:
        raise ValueError('Protected base model hash mismatch')
    prev = json.loads((POST.parent/'result.json').read_text(encoding='utf-8'))
    if sha(POST) != prev['candidate_sha256'] or any(
        sha(PAIR/f'layer-{i}.pt') != prev['upstream_pair_sha256'][str(i)] for i in (6, 12)
    ):
        raise ValueError('Candidate provenance mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    base.requires_grad_(False)
    original = {i: base.model.layers[i] for i in (6, 12, 18)}
    replacements = {i: _load_candidate(PAIR/f'layer-{i}.pt', i, base).eval() for i in (6, 12)}
    replacements[18] = _load_candidate(POST, 18, base).eval()
    prefixes = [closed_think_inputs(tokenizer, case['prompt'], 'cuda') for case in CASES]
    for prefix in prefixes:
        if not 128 < prefix['input_ids'].shape[1] <= 220:
            raise ValueError(f"Expected prompt length between 129 and 220: {prefix['input_ids'].shape[1]}")
    short = closed_think_inputs(tokenizer, '你好，请简单介绍一下你自己。', 'cuda')
    for index in (6, 12, 18):
        base.model.layers[index] = replacements[index]
    with torch.inference_mode():
        short_before = base(**short, use_cache=False).logits[:, -1].float()
    for block in replacements.values():
        block.extend_positions(256)
    with torch.inference_mode():
        short_after = base(**short, use_cache=False).logits[:, -1].float()
    short_delta = float((short_before-short_after).abs().max())
    results = {}
    for mode, indices in (('original', ()), ('extended_pair', (6, 12)), ('extended_triple', (6, 12, 18))):
        for index in (6, 12, 18):
            base.model.layers[index] = replacements[index] if index in indices else original[index]
        rows = []
        for case, prefix in zip(CASES, prefixes):
            metrics = answer_metrics(base, tokenizer, prefix, case['reference'])
            rows.append({'name': case['name'], 'prompt': case['prompt'], 'reference': case['reference'],
                         'prompt_tokens': prefix['input_ids'].shape[1], **metrics})
        results[mode] = rows
        print(json.dumps({'mode': mode, 'reference_nll': [x['reference_mean_nll'] for x in rows]},
                         ensure_ascii=False), flush=True)
    report = {'results': results, 'extended_length': 256, 'old_length': 128,
              'new_position_initialization': 'zero', 'short_max_logit_delta_after_extension': short_delta,
              'base_sha256': WEIGHT_SHA, 'pair_sha256': {str(i): sha(PAIR/f'layer-{i}.pt') for i in (6, 12)},
              'layer18_sha256': sha(POST), 'elapsed_seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Two manually authored long prompts; memory reader not active.',
                              'New position embeddings are untrained zeros, not learned extrapolation.',
                              'NLL is a narrow teacher-forced metric, not open-chat acceptance.',
                              'Generation capped at 24 tokens with a closed reasoning prefix.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    run()
