"""Fit a supervised read-to-embedding probe and test held value tokens.

This diagnostic uses whole-fact writes and no test-cohort normalization.
It is not a generation adapter and does not update the language model.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F
from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_ttt_ntp import attach_read_adapters, install, use_memory

OUT = ROOT / 'runs/deepseek-spora-ttt-embedding-probe-20261114.json'
SOURCE = crossed.OUT
TRAIN_VALUES = ('酸奶', '豆浆', '咖啡', '果汁', '可乐', '牛奶', '红茶', '绿茶',
                '汽水', '米酒', '啤酒', '柠檬水')
HELD_VALUES = ('奶茶', '椰汁', '橙汁', '矿泉水')
RIDGES = (0.01, 0.1, 1.0)


def fit_predict(source_features, source_targets, target_features, ridge):
    """Dual ridge regression with only source-fitted centering and scaling."""
    source = source_features.float()
    target = target_features.float()
    mean = source.mean(0, keepdim=True)
    scale = (source - mean).square().sum(-1).mean().sqrt().clamp_min(1e-12)
    source = (source - mean) / scale
    target = (target - mean) / scale
    target_mean = source_targets.float().mean(0, keepdim=True)
    gram = source @ source.T
    regularized = gram + ridge * torch.eye(gram.shape[0], device=gram.device)
    coefficients = torch.linalg.solve(regularized, source_targets.float() - target_mean)
    return target @ source.T @ coefficients + target_mean


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    meta = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_sha = p.sha(SOURCE / 'adapters.pt')
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    payload = torch.load(SOURCE / 'adapters.pt', map_location='cpu', weights_only=True)
    if adapter_sha != meta['adapter_sha256'] or base_sha != p.WEIGHT_SHA or payload['base_sha256'] != base_sha:
        raise ValueError('Candidate provenance mismatch')
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
    values = TRAIN_VALUES + HELD_VALUES
    ids = [tokenizer.encode(value, add_special_tokens=False) for value in values]
    if any(not token_ids for token_ids in ids) or len({token_ids[0] for token_ids in ids}) != len(ids):
        raise ValueError('Values need distinct nonempty first tokens')
    token_ids = [token_ids[0] for token_ids in ids]
    embedding = model.model.embed_tokens.weight.detach()
    target_vectors = F.normalize(embedding[torch.tensor(token_ids, device='cuda')].float(), dim=-1).cpu()
    features = []
    for name in crossed.TRAIN_NAMES + crossed.HELD_NAMES:
        prompt = pilot.closed_think_inputs(tokenizer, crossed.TEMPLATES[0].format(name=name), 'cuda')
        person_features = []
        for value in values:
            memory = pilot.device_memory(pilot.make_memory(model, tokenizer, wrappers, f'{name}喜欢喝{value}。'))
            captured = {}
            handles = []
            for layer, adapter in adapters.items():
                def capture(module, args, output, layer=layer):
                    captured[layer] = args[0][0, -1].detach().float().cpu()
                handles.append(adapter.register_forward_hook(capture))
            try:
                with torch.inference_mode(), use_memory(memory):
                    model(**prompt, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            if set(captured) != set(adapters):
                raise ValueError('Missing memory read')
            person_features.append(torch.cat([captured[layer] for layer in sorted(captured)]))
        features.append(torch.stack(person_features))
        print(json.dumps({'prepared': name, 'values': len(values)}, ensure_ascii=False), flush=True)
    features = torch.stack(features)
    source_count = len(crossed.TRAIN_NAMES)
    train_count = len(TRAIN_VALUES)
    source_features = features[:source_count, :train_count].reshape(-1, features.shape[-1])
    source_targets = target_vectors[:train_count].repeat(source_count, 1)
    target_features = features[source_count:].reshape(-1, features.shape[-1])
    expected_ids = torch.tensor(token_ids * len(crossed.HELD_NAMES), dtype=torch.long)
    target_values = list(values) * len(crossed.HELD_NAMES)
    rows = []
    for ridge in RIDGES:
        predicted = F.normalize(fit_predict(source_features, source_targets, target_features, ridge), dim=-1).to('cuda')
        maxima = torch.full((predicted.shape[0],), float('-inf'), device='cuda')
        winners = torch.zeros(predicted.shape[0], dtype=torch.long, device='cuda')
        target_scores = (predicted * F.normalize(embedding.index_select(0, expected_ids.to('cuda')).float(), dim=-1)).sum(-1)
        ranks = torch.ones(predicted.shape[0], dtype=torch.long, device='cuda')
        for start_index in range(0, embedding.shape[0], 4096):
            table = F.normalize(embedding[start_index:start_index+4096].float(), dim=-1)
            scores = predicted @ table.T
            maximum, indices = scores.max(-1)
            better = maximum > maxima
            winners = torch.where(better, indices + start_index, winners)
            maxima = torch.maximum(maxima, maximum)
            ranks += (scores > target_scores[:, None]).sum(-1)
        restricted = predicted @ target_vectors.to('cuda').T
        restricted_top = restricted.argmax(-1).cpu().tolist()
        group_results = []
        for index, value in enumerate(target_values):
            is_held = value in HELD_VALUES
            group_results.append({'name': crossed.HELD_NAMES[index // len(values)], 'value': value,
                                  'group': 'held_value' if is_held else 'seen_value',
                                  'target_first_token_id': int(expected_ids[index]),
                                  'value_token_count': len(ids[index % len(values)]),
                                  'full_vocab_rank': int(ranks[index]),
                                  'full_vocab_winner_id': int(winners[index]),
                                  'restricted_winner': values[restricted_top[index]]})
        rows.append({'ridge': ridge, 'cases': group_results,
                     'summary': {group: {'cases': sum(r['group'] == group for r in group_results),
                                         'full_top1': sum(r['full_vocab_rank'] == 1 for r in group_results if r['group'] == group),
                                         'full_top10': sum(r['full_vocab_rank'] <= 10 for r in group_results if r['group'] == group),
                                         'restricted_top1': sum(r['restricted_winner'] == r['value'] for r in group_results if r['group'] == group)}
                                 for group in ('seen_value', 'held_value')}})
        print(json.dumps({'ridge': ridge, 'summary': rows[-1]['summary']}, ensure_ascii=False), flush=True)
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(SOURCE / 'adapters.pt') != adapter_sha:
        raise ValueError('Protected source changed')
    report = {'status': 'whole_fact_read_to_embedding_probe', 'rows': rows,
              'train_values': TRAIN_VALUES, 'held_values': HELD_VALUES,
              'source_names': crossed.TRAIN_NAMES, 'held_names': crossed.HELD_NAMES,
              'base_sha256': base_sha, 'adapter_sha256': adapter_sha,
              'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Ridge uses labeled token embeddings for 12 known values and 4 training names.',
                              'Full vocabulary evaluates first token only; multi-token output is not generated.',
                              'Restricted comparison includes the held values as oracle candidates and is not open vocabulary.',
                              'Single authored fact and question template; held values were exposed to prior development, not sealed.',
                              'Ridge alternatives are development comparisons, not independently selected hyperparameters.',
                              'Per-case fast states and addresses are manually supplied, no natural routing or persistence.']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'completed': True}), flush=True)


if __name__ == '__main__':
    main()
