"""Reload online adapters and check unseen non-memory and unknown prompts."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_online_lora_memory_pilot as online
from scripts import deepseek_spora_online_qa_replay_pilot as replay
from scripts import deepseek_spora_online_qa_guard_pilot as guard
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed

OUT = ROOT / 'runs/deepseek-spora-online-qa-independent-audit-20261114.json'
PROMPTS = (
    ('generic', '请用一句话描述雨后的天空。'),
    ('generic', '三加二等于几？'),
    ('generic', '我今天有点疲倦，给我一个休息建议。'),
    ('unknown', '小王现在喜欢哪种饮料？只说名称。'),
    ('unknown', '请回答小雪爱喝什么饮品。'),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {'replay': replay.OUT / 'adapters.pt', 'guard': guard.OUT / 'adapter.pt'}
    reports = {'replay': json.loads((replay.OUT / 'result.json').read_text(encoding='utf-8')),
               'guard': json.loads((guard.OUT / 'result.json').read_text(encoding='utf-8'))}
    hashes = {key: p.sha(path) for key, path in paths.items()}
    for key in paths:
        if hashes[key] != reports[key]['adapter_sha256'] or reports[key]['base_sha256'] != p.WEIGHT_SHA:
            raise ValueError('Candidate lineage mismatch')
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Base digest mismatch')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    modules = []
    for layer in online.LAYERS:
        mlp = model.model.layers[layer].mlp
        module = online.OnlineDelta(mlp.down_proj, online.RANK)
        mlp.down_proj = module
        modules.append(module)
    payloads = {key: torch.load(path, map_location='cpu', weights_only=True) for key, path in paths.items()}
    rows = []
    for mode in ('base', 'replay', 'guard'):
        for module in modules:
            module.enabled = mode != 'base'
        if mode != 'base':
            weights = payloads[mode]['arms']['replay'] if mode == 'replay' else payloads[mode]['adapters']
            for module, state in zip(modules, weights):
                missing = module.load_state_dict(state, strict=False)
                if missing.unexpected_keys or set(missing.missing_keys) != {'base.weight'}:
                    raise ValueError('Unexpected adapter state')
        for group, question in PROMPTS:
            answer = pilot.generate(model, tokenizer, question)
            rows.append({'mode': mode, 'group': group, 'question': question, 'answer': answer,
                         'contains_fact_value': any(value in answer['answer'] for _, value in replay.EVENTS)})
        print(json.dumps({'mode': mode, 'completed': len(rows)}), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA or any(p.sha(paths[key]) != hashes[key] for key in paths):
        raise ValueError('Protected source changed')
    summary = {}
    for mode in ('replay', 'guard'):
        subset = [row for row in rows if row['mode'] == mode]
        generic = [row for row in subset if row['group'] == 'generic']
        unknown = [row for row in subset if row['group'] == 'unknown']
        summary[mode] = {'generic_cases': len(generic),
                         'generic_token_matches_base': sum(row['answer']['token_ids'] == next(
                             base['answer']['token_ids'] for base in rows if base['mode'] == 'base'
                             and base['question'] == row['question']) for row in generic),
                         'generic_contains_fact_value': sum(row['contains_fact_value'] for row in generic),
                         'unknown_cases': len(unknown),
                         'unknown_contains_fact_value': sum(row['contains_fact_value'] for row in unknown)}
    report = {'status': 'online_qa_held_prompt_audit', 'summary': summary, 'rows': rows,
              'base_sha256': p.WEIGHT_SHA, 'adapter_sha256': hashes,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Three generic and two unknown authored prompts only.',
                              'String matching can miss erroneous claims or flag benign mentions.',
                              'Token equality does not measure semantic quality.',
                              'Both candidates use supervised synthetic QA and program-selected replay.',
                              'No true user data, durable individual, broad chat benchmark, or history query.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
