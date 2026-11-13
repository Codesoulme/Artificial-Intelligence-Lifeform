"""Reload selective-write candidate and check counterfactual and empty states."""
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_selective_train_pilot import OUT as SOURCE
from scripts.deepseek_spora_ttt_selective_write_audit import selected_memory
from spora.research_ttt_ntp import attach_read_adapters, fresh_memory, install

OUT = ROOT / 'runs/deepseek-spora-ttt-selective-reload-20261114.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    candidate_sha = p.sha(SOURCE / 'adapters.pt')
    if source['adapter_sha256'] != candidate_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Candidate provenance mismatch')
    payload = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    if payload.get('base_sha256') != p.WEIGHT_SHA or source.get('write_mode') != 'oracle_value_token_positions':
        raise ValueError('Unexpected candidate metadata')
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
    rows, empty_checks = [], []
    for name in crossed.HELD_NAMES:
        question = crossed.TEMPLATES[0].format(name=name)
        off = pilot.generate(model, tokenizer, question)
        empty = pilot.generate(model, tokenizer, question, fresh_memory(wrappers))
        empty_checks.append({'name': name, 'off': off, 'empty': empty,
                             'tokens_match': off['token_ids'] == empty['token_ids']})
        for value in crossed.TRAIN_VALUES:
            memory, positions = selected_memory(model, tokenizer, wrappers, f'{name}喜欢喝{value}。', value)
            answer = pilot.generate(model, tokenizer, question, memory)
            old = next(row for row in source['rows'] if row['group'] == 'held_name'
                       and row['name'] == name and row['value'] == value)
            rows.append({'name': name, 'value': value, 'answer': answer, 'positions': positions,
                         'exact': crossed.exact(answer['answer'], value),
                         'token_match': answer['token_ids'] == old['answer']['token_ids']})
        print(json.dumps({'name': name, 'completed': len(rows)}, ensure_ascii=False), flush=True)
    pairs = []
    for name in crossed.HELD_NAMES:
        for left, right in itertools.combinations([row for row in rows if row['name'] == name], 2):
            pairs.append({'name': name, 'values': [left['value'], right['value']],
                          'both_correct': left['exact'] and right['exact'],
                          'changed': left['answer']['token_ids'] != right['answer']['token_ids']})
    if p.sha(SOURCE / 'adapters.pt') != candidate_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected input changed')
    summary = {'cases': len(rows), 'exact': sum(row['exact'] for row in rows),
               'token_matches': sum(row['token_match'] for row in rows),
               'pairs': len(pairs), 'both_correct': sum(pair['both_correct'] for pair in pairs),
               'changed': sum(pair['changed'] for pair in pairs),
               'empty_checks': len(empty_checks), 'empty_matches': sum(row['tokens_match'] for row in empty_checks)}
    result = {'status': 'oracle_selected_reload_audit', 'summary': summary, 'rows': rows,
              'pairs': pairs, 'empty_checks': empty_checks, 'adapter_sha256': candidate_sha,
              'base_sha256': p.WEIGHT_SHA, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Value locations supplied from the known fact; no natural selection.',
                              'Two held names and four trained values; pair results overlap.',
                              'Empty state equivalence does not prove correct unknown-memory refusal.',
                              'Reconstructs temporary fast weights, not a persisted individual.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
