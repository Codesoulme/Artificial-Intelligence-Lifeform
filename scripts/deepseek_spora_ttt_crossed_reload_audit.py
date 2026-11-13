"""Reload crossed-value candidate and verify paired counterfactual answers."""
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as experiment
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import attach_read_adapters, install

SOURCE = experiment.OUT
OUT = ROOT / 'runs/deepseek-spora-ttt-crossed-reload-20261114.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    prior = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(SOURCE / 'adapters.pt')
    if adapter_sha != prior['adapter_sha256'] or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Source digest mismatch')
    payload = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    if payload['base_sha256'] != p.WEIGHT_SHA or payload['format'] != 'spora-ttt-crossed-v1':
        raise ValueError('Unexpected checkpoint format')
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer, adapter in adapters.items():
        adapter.load_state_dict(payload['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)
    rows = []
    for name in experiment.HELD_NAMES:
        for value in experiment.TRAIN_VALUES:
            case = pilot.build_case(model, tokenizer, wrappers, name, value, experiment.TEMPLATES[0])
            generated = pilot.generate(model, tokenizer, case['question'], pilot.device_memory(case['memory']))
            reference = next(row for row in prior['rows'] if row['group'] == 'held_name'
                             and row['name'] == name and row['value'] == value)
            rows.append({'name': name, 'value': value, 'answer': generated,
                         'exact': experiment.exact(generated['answer'], value),
                         'token_match': generated['token_ids'] == reference['answer']['token_ids']})
        print(json.dumps({'name': name, 'completed': len(rows)}, ensure_ascii=False), flush=True)
    pairs = []
    for name in experiment.HELD_NAMES:
        subset = [row for row in rows if row['name'] == name]
        for left, right in itertools.combinations(subset, 2):
            pairs.append({'name': name, 'values': [left['value'], right['value']],
                          'both_correct': left['exact'] and right['exact'],
                          'answer_changed': left['answer']['token_ids'] != right['answer']['token_ids']})
    if p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA or p.sha(SOURCE / 'adapters.pt') != adapter_sha:
        raise ValueError('Protected file changed')
    summary = {'cases': len(rows), 'exact': sum(row['exact'] for row in rows),
               'token_matches': sum(row['token_match'] for row in rows),
               'pairs': len(pairs), 'both_correct': sum(pair['both_correct'] for pair in pairs),
               'answer_changed': sum(pair['answer_changed'] for pair in pairs)}
    report = {'status': 'crossed_candidate_reload_audit', 'summary': summary, 'rows': rows,
              'pairs': pairs, 'adapter_sha256': adapter_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Recreates temporary fast weights from the same fact text; no saved individual reload.',
                              f'{len(experiment.HELD_NAMES)} held names and {len(experiment.TRAIN_VALUES)} trained values; pairs overlap.',
                              'Explicit one-fact writes and supplied questions, not autonomous routing.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
