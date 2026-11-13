"""Reload unknown-supervised adapter and inspect held prompt tradeoffs."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_online_lora_memory_pilot as online
from scripts import deepseek_spora_online_unknown_training_pilot as candidate
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-online-unknown-reload-audit-20261114.json'
PROMPTS = (
    ('known', '咖啡', '小林现在爱喝哪一种饮品？'),
    ('known', '豆浆', '阿宁目前喜欢什么饮料？'),
    ('unknown', None, '小林华现在爱喝哪一种饮品？'),
    ('unknown', None, '阿宁悦目前喜欢什么饮料？'),
    ('unknown', None, '小王现在爱喝哪一种饮品？'),
    ('unknown', None, '小雪目前喜欢什么饮料？'),
    ('generic', None, '请用一句话描述雨后的天空。'),
    ('generic', None, '三加二等于几？'),
    ('generic', None, '我今天有点疲倦，给我一个休息建议。'),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads((candidate.OUT / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(candidate.OUT / 'adapter.pt')
    if meta['adapter_sha256'] != adapter_sha or meta['base_sha256'] != p.WEIGHT_SHA:
        raise ValueError('Candidate digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    payload = torch.load(candidate.OUT / 'adapter.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != p.WEIGHT_SHA or payload['format'] != 'online-unknown-training-v1':
        raise ValueError('Candidate metadata mismatch')
    for layer, state in zip(payload['layers'], payload['adapters']):
        mlp = model.model.layers[layer].mlp
        module = online.OnlineDelta(mlp.down_proj, payload['rank'])
        mlp.down_proj = module
        missing = module.load_state_dict(state, strict=False)
        if missing.unexpected_keys or set(missing.missing_keys) != {'base.weight'}:
            raise ValueError('Unexpected adapter state')
        modules.append(module)
    rows = []
    for mode in ('base', 'candidate'):
        for module in modules:
            module.enabled = mode == 'candidate'
        for group, expected, question in PROMPTS:
            answer = pilot.generate(model, tokenizer, question)
            rows.append({'mode': mode, 'group': group, 'expected': expected, 'question': question,
                         'answer': answer, 'explicit_unknown': candidate.refusal(answer['answer']),
                         'known_exact': expected is not None and answer['answer'].strip().rstrip('。.!！ ') == expected,
                         'contains_observed_value': any(value in answer['answer'] for _, value in
                                                        (('小林', '酸奶'), ('阿宁', '豆浆'), ('小林', '咖啡')))})
        print(json.dumps({'mode': mode, 'completed': len(rows)}), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA or p.sha(candidate.OUT / 'adapter.pt') != adapter_sha:
        raise ValueError('Protected source changed')
    subset = [row for row in rows if row['mode'] == 'candidate']
    summary = {'known_exact': sum(row['known_exact'] for row in subset if row['group'] == 'known'),
               'known_cases': sum(row['group'] == 'known' for row in subset),
               'known_refusals': sum(row['explicit_unknown'] for row in subset if row['group'] == 'known'),
               'unknown_explicit': sum(row['explicit_unknown'] for row in subset if row['group'] == 'unknown'),
               'unknown_cases': sum(row['group'] == 'unknown' for row in subset),
               'unknown_value_mentions': sum(row['contains_observed_value'] for row in subset if row['group'] == 'unknown'),
               'generic_matches_base': sum(row['answer']['token_ids'] == next(
                   base['answer']['token_ids'] for base in rows if base['mode'] == 'base'
                   and base['question'] == row['question']) for row in subset if row['group'] == 'generic'),
               'generic_cases': sum(row['group'] == 'generic' for row in subset)}
    report = {'status': 'unknown_training_held_prompt_audit', 'summary': summary, 'rows': rows,
              'base_sha256': p.WEIGHT_SHA, 'adapter_sha256': adapter_sha,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Nine authored development prompts and a single trained candidate.',
                              'Names and answer forms may overlap earlier project development.',
                              'Exact refusal scoring misses other correct uncertainty expressions.',
                              'Observed-value substring is not a complete hallucination or quality metric.',
                              'No new facts, natural write, historical recall, durable state or real user.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
