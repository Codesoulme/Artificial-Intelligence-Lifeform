"""Contrast prompted base classification with frozen feature heads, offline only."""
import gc
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_spora_relation_gate_pilot import OUT as V1

V2 = ROOT / 'runs/deepseek-spora-relation-gate-v2-pilot-20261114'
OUTPUT = ROOT / 'runs/deepseek-spora-relation-base-comparison-20261114.json'
NEW_CASES = (
    ('read', '能回忆一下小岚喜欢喝的饮品名称吗？', 1),
    ('read', '记录中阿明的饮料偏好是什么？', 1),
    ('attribute', '小岚喜欢的饮料是甜的还是酸的？', 0),
    ('attribute', '阿明偏爱的饮品装在什么容器里？', 0),
    ('negative', '千万别透露小岚喜欢喝的饮料。', 0),
    ('negative', '不要说阿明爱喝什么。', 0),
    ('write', '把小岚的饮料记录清除。', 0),
    ('write', '记下阿明爱喝果汁。', 0),
    ('hypothesis', '假如小岚喝茶，写句描写。', 0),
    ('hypothesis', '不查记录，猜阿明爱喝什么。', 0),
    ('recommend', '给小岚推荐解渴的饮品。', 0),
    ('ordinary', '请说明一周有多少天。', 0),
)


def prompt_for(question):
    return ('判断下面的请求是否要求查询某个人喜欢的饮料名称。'
            '如果询问颜色、价格、味道，要求修改、猜测、推荐或不告诉，则不是。'
            '只回答“是”或“否”。\n请求：' + question)


def classify(model, tokenizer, rows, variant):
    token_ids = {word: tokenizer.encode(word, add_special_tokens=False) for word in ('是', '否')}
    if any(len(ids) != 1 for ids in token_ids.values()) or token_ids['是'] == token_ids['否']:
        raise ValueError('Expected two distinct single-token labels')
    for i, row in enumerate(rows):
        inputs = closed_think_inputs(tokenizer, prompt_for(row['question']), 'cuda')
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True).logits[0, -1].float()
        yes, no = float(logits[token_ids['是'][0]]), float(logits[token_ids['否'][0]])
        row[variant] = {'yes_logit': yes, 'no_logit': no, 'accepted': yes > no,
                        'top_token_id': int(logits.argmax()),
                        'top_token': tokenizer.decode([int(logits.argmax())])}
        if (i + 1) % 10 == 0:
            print(json.dumps({'variant': variant, 'done': i + 1, 'total': len(rows)}), flush=True)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    old_results = []
    for directory in (V1, V2):
        report = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        if report['base_sha256'] != base_sha or p.sha(directory / 'gate.pt') != report['gate_sha256']:
            raise ValueError('Head provenance mismatch')
        old_results.append(report)
    prior = [{k: r[k] for k in ('group', 'question', 'label')} for r in old_results[0]['rows'] if r['split'] == 'test']
    prior2 = [r for r in old_results[1]['rows'] if r['split'] == 'test']
    if [r['question'] for r in prior] != [r['question'] for r in prior2]:
        raise ValueError('Development rows differ')
    for i, row in enumerate(prior):
        row['set'] = 'reused_development'
        row['head_v1'] = old_results[0]['rows'][next(j for j, x in enumerate(old_results[0]['rows'])
                                                    if x['question'] == row['question'])]['accepted']
        row['head_v2'] = prior2[i]['accepted']
    rows = prior + [{'set': 'new_authored', 'group': group, 'question': question, 'label': label}
                    for group, question, label in NEW_CASES]
    if len({r['question'] for r in rows}) != len(rows):
        raise ValueError('Repeated comparison questions')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    original = p.load_full_streamed().eval()
    original.requires_grad_(False)
    classify(original, tokenizer, rows, 'original_base')
    del original
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='spora-relation-baseline-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        classify(chat.base, tokenizer, rows, 'modified_base')
        if p.frozen_digest(chat.organism, chat.state) != original_digest:
            raise ValueError('Original individual changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected file changed')
    summary = {}
    for split in ('reused_development', 'new_authored'):
        summary[split] = {}
        for group in sorted({r['group'] for r in rows if r['set'] == split}):
            subset = [r for r in rows if r['set'] == split and r['group'] == group]
            scores = {'cases': len(subset)}
            for variant in ('original_base', 'modified_base'):
                scores[variant] = {'accepted': sum(r[variant]['accepted'] for r in subset),
                                   'label_top': sum(r[variant]['top_token'] in ('是', '否') for r in subset)}
            if split == 'reused_development':
                scores['head_v1'] = sum(r['head_v1'] for r in subset)
                scores['head_v2'] = sum(r['head_v2'] for r in subset)
            summary[split][group] = scores
    report = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'head_sha256': [r['gate_sha256'] for r in old_results], 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Single forced-choice prompt; answer scores are not calibrated probabilities.',
                              'Top vocabulary token may be neither label; forced-choice is not autonomous output.',
                              'Comparison isolates classification, not actual natural routing or memory use.',
                              'Reused development set was used for model selection; new hand-authored cases may overlap earlier work.',
                              'Prompt may be vulnerable to adversarial instructions; no access-control claim.',
                              'Original and modified base have different feature paths; no proof of a unique cause.']}
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
