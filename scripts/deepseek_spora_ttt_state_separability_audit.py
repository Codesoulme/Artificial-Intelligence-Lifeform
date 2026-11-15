"""Separate TTT write information from language-facing read failures.

This is a read-only diagnostic.  It reuses the crossed-value TTT candidate,
captures the written fast weights and the two language-facing representations,
and evaluates a small probe fitted only on training names and values.  It does
not modify the base checkpoint or the candidate adapter.
"""
import itertools
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


SOURCE = crossed.OUT
OUT = ROOT / 'runs/deepseek-spora-ttt-state-separability-audit-20261116.json'
LAYERS = (6, 12)
RIDGES = (0.01, 0.1, 1.0)


def cosine(left, right):
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    return float(F.cosine_similarity(left, right, dim=0))


def representation(record, kind):
    if kind == 'state':
        return {layer: record['state'][layer].float() for layer in LAYERS}
    if kind == 'raw':
        return {layer: record['raw'][layer].float() for layer in LAYERS}
    if kind == 'adapted':
        return {layer: record['adapted'][layer].float() for layer in LAYERS}
    raise ValueError(f'Unknown representation: {kind}')


def pairwise_summary(records, kind):
    pairs = []
    for left, right in itertools.combinations(records, 2):
        if left['name'] == right['name'] and left['value'] != right['value']:
            relation = 'different_value_same_name'
        elif left['value'] == right['value'] and left['name'] != right['name']:
            relation = 'same_value_cross_name'
        else:
            continue
        left_rep = representation(left, kind)
        right_rep = representation(right, kind)
        layers = {}
        for layer in LAYERS:
            x = left_rep[layer]
            y = right_rep[layer]
            layers[str(layer)] = {
                'cosine': cosine(x, y),
                'difference_norm': float((x - y).norm()),
                'left_norm': float(x.norm()),
                'right_norm': float(y.norm()),
            }
        pairs.append({'relation': relation, 'left': [left['name'], left['value']],
                      'right': [right['name'], right['value']], 'layers': layers})
    summary = {}
    for relation in ('different_value_same_name', 'same_value_cross_name'):
        subset = [row for row in pairs if row['relation'] == relation]
        summary[relation] = {'pairs': len(subset), 'layers': {}}
        for layer in LAYERS:
            values = [row['layers'][str(layer)] for row in subset]
            summary[relation]['layers'][str(layer)] = {
                'mean_cosine': sum(row['cosine'] for row in values) / len(values),
                'mean_difference_norm': sum(row['difference_norm'] for row in values) / len(values),
            }
    return summary, pairs


def fit_ridge(source_features, source_targets, target_features, ridge):
    source = source_features.float()
    target = target_features.float()
    mean = source.mean(0, keepdim=True)
    scale = (source - mean).square().mean().sqrt().clamp_min(1e-6)
    source = (source - mean) / scale
    target = (target - mean) / scale
    target_mean = source_targets.float().mean(0, keepdim=True)
    gram = source @ source.T
    system = gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    coefficients = torch.linalg.solve(system, source_targets.float() - target_mean)
    return target @ source.T @ coefficients + target_mean


def run_probe(records, model, tokenizer, kind, ridge):
    values = crossed.TRAIN_VALUES + crossed.HELD_VALUES
    token_ids = [tokenizer.encode(value, add_special_tokens=False)[0] for value in values]
    embedding = F.normalize(model.model.embed_tokens.weight.detach().float().cpu(), dim=-1)
    target_vectors = embedding[torch.tensor(token_ids)]

    def feature(record):
        rep = representation(record, kind)
        return torch.cat([rep[layer].reshape(-1) for layer in LAYERS])

    source = [row for row in records if row['name'] in crossed.TRAIN_NAMES
              and row['value'] in crossed.TRAIN_VALUES]
    test = [row for row in records if row['name'] in crossed.HELD_NAMES]
    source_features = torch.stack([feature(row) for row in source])
    source_targets = target_vectors[torch.tensor([values.index(row['value']) for row in source])]
    target_features = torch.stack([feature(row) for row in test])
    predicted = F.normalize(fit_ridge(source_features, source_targets, target_features, ridge), dim=-1)
    scores = predicted @ target_vectors.T
    winners = scores.argmax(-1)
    rows = []
    for index, row in enumerate(test):
        expected = values.index(row['value'])
        rows.append({'name': row['name'], 'value': row['value'],
                     'group': 'held_name_seen_value' if row['value'] in crossed.TRAIN_VALUES
                     else 'held_name_unseen_value',
                     'restricted_rank': int((scores[index] > scores[index, expected]).sum()) + 1,
                     'restricted_winner': values[int(winners[index])],
                     'target_similarity': float(scores[index, expected])})
    return {'representation': kind, 'ridge': ridge, 'rows': rows,
            'summary': {group: {
                'cases': sum(row['group'] == group for row in rows),
                'top1': sum(row['group'] == group and row['restricted_rank'] == 1 for row in rows),
                'top3': sum(row['group'] == group and row['restricted_rank'] <= 3 for row in rows),
            } for group in ('held_name_seen_value', 'held_name_unseen_value')}}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    metadata = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    adapter_path = SOURCE / 'adapters.pt'
    adapter_sha = p.sha(adapter_path)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    payload = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if (metadata['adapter_sha256'] != adapter_sha or base_sha != p.WEIGHT_SHA
            or payload['base_sha256'] != base_sha):
        raise ValueError('Candidate or base provenance mismatch')

    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    adapters = attach_read_adapters(wrappers, rank=payload['rank'])
    for layer, adapter in adapters.items():
        adapter.load_state_dict(payload['adapters'][str(layer)], strict=True)
    model.requires_grad_(False)

    records = []
    values = crossed.TRAIN_VALUES + crossed.HELD_VALUES
    for name in crossed.TRAIN_NAMES + crossed.HELD_NAMES:
        question = crossed.TEMPLATES[0].format(name=name)
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        for value in values:
            case = pilot.build_case(model, tokenizer, wrappers, name, value,
                                    crossed.TEMPLATES[0])
            captured = {}
            hooks = []
            for layer, adapter in adapters.items():
                def capture(module, args, output, layer=layer):
                    captured[layer] = {
                        'raw': args[0][0, -1].detach().float().cpu(),
                        'adapted': output[0, -1].detach().float().cpu(),
                    }
                hooks.append(adapter.register_forward_hook(capture))
            try:
                with torch.inference_mode(), use_memory(pilot.device_memory(case['memory'])):
                    logits = model(**prompt, use_cache=False, return_dict=True).logits[0, -1]
            finally:
                for hook in hooks:
                    hook.remove()
            if set(captured) != set(LAYERS):
                raise ValueError('Missing read representation')
            token_id = tokenizer.encode(value, add_special_tokens=False)[0]
            rank = int((logits.float() > logits.float()[token_id]).sum()) + 1
            records.append({
                'name': name, 'value': value,
                'group': 'train_name' if name in crossed.TRAIN_NAMES else 'held_name',
                'value_group': 'seen_value' if value in crossed.TRAIN_VALUES else 'unseen_value',
                'target_first_token_rank': rank,
                'winner_token': tokenizer.decode([int(logits.argmax())], skip_special_tokens=False),
                'state': {layer: case['memory'][layer].detach().float().cpu() for layer in LAYERS},
                'raw': {layer: captured[layer]['raw'] for layer in LAYERS},
                'adapted': {layer: captured[layer]['adapted'] for layer in LAYERS},
            })
            print(json.dumps({'name': name, 'value': value, 'target_rank': rank},
                             ensure_ascii=False), flush=True)

    pairwise = {}
    pair_rows = {}
    for kind in ('state', 'raw', 'adapted'):
        summary, pairs = pairwise_summary(records, kind)
        pairwise[kind] = summary
        pair_rows[kind] = pairs

    probes = []
    for kind in ('raw', 'adapted'):
        for ridge in RIDGES:
            probe = run_probe(records, model, tokenizer, kind, ridge)
            probes.append(probe)
            print(json.dumps({'probe': kind, 'ridge': ridge,
                              'summary': probe['summary']}, ensure_ascii=False), flush=True)

    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(adapter_path) != adapter_sha:
        raise ValueError('Protected file changed')
    result = {
        'status': 'ttt_state_separability_audit',
        'source': str(SOURCE.relative_to(ROOT)),
        'base_sha256': base_sha, 'adapter_sha256': adapter_sha,
        'layers': list(LAYERS), 'values': list(values),
        'records': [{key: value for key, value in row.items()
                     if key not in ('state', 'raw', 'adapted')} for row in records],
        'pairwise_summary': pairwise,
        'pairwise_details': pair_rows,
        'probes': probes,
        'seconds': time.perf_counter() - started,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': [
            'Facts, names, questions, and fast-state construction are program-selected.',
            'The probe uses a small finite candidate vocabulary and is not a chat test.',
            'A separable fast state does not prove natural-language routing or persistence.',
            'The write target is still the current untrained TTT-NTP target; this audit does not optimize it.',
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                   encoding='utf-8')
    print(json.dumps({'status': result['status'], 'seconds': result['seconds']}, ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
