"""Compare learned token reconstruction from shallow and final decoder states.

This is an oracle-span representation experiment, not an autonomous writer.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F

from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import atomic_save
from spora.semantic_text_writer import TextSemanticWriter, value_positions

OUT = ROOT / 'runs/deepseek-spora-writer-layer-control-20261116'
SEEDS = (93471, 93472)
BATCH_SIZE = 8


def create_cases(train_names, held_names, train_values, held_values):
    if (not all((train_names, held_names, train_values, held_values))
            or set(train_names) & set(held_names) or set(train_values) & set(held_values)
            or any(len(set(values)) != len(values) for values in
                   (train_names, held_names, train_values, held_values))):
        raise ValueError('Name and value splits must be nonempty and disjoint')
    return [{'name': name, 'value': value, 'name_split': name_split, 'value_split': value_split}
            for name_split, names in (('train', train_names), ('held', held_names))
            for value_split, values in (('train', train_values), ('held', held_values))
            for name in names for value in values]


def collect(model, tokenizer, cases):
    prepared = []
    for case in cases:
        fact = f"{case['name']}喜欢喝{case['value']}。"
        encoded, positions = value_positions(tokenizer, fact, case['value'])
        if len(positions) != 1:
            raise ValueError('Expected a single aligned value token')
        ids = encoded['input_ids'][0]
        prepared.append((ids, positions[0]))
    shallow, final, targets = [], [], []
    for start in range(0, len(prepared), BATCH_SIZE):
        batch = prepared[start:start + BATCH_SIZE]
        length = max(ids.numel() for ids, _ in batch)
        ids = torch.full((len(batch), length), tokenizer.pad_token_id, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        positions = torch.tensor([position for _, position in batch], device='cuda')
        for row, (tokens, _) in enumerate(batch):
            ids[row, :tokens.numel()] = tokens.to('cuda')
            mask[row, :tokens.numel()] = 1
        rows = torch.arange(len(batch), device='cuda')
        with torch.inference_mode():
            result = model.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                 output_hidden_states=True, return_dict=True)
            shallow.append(result.hidden_states[1][rows, positions].float().cpu())
            final.append(result.last_hidden_state[rows, positions].float().cpu())
            targets.append(model.model.embed_tokens(ids[rows, positions]).float().cpu())
        if (start // BATCH_SIZE + 1) % 40 == 0:
            print(json.dumps({'prepared': start + len(batch), 'total': len(cases)}), flush=True)
    return {'shallow_layer_1': torch.cat(shallow), 'final_normalized': torch.cat(final)}, torch.cat(targets)


def score(writer, features, targets, cases, candidate_values, candidates):
    with torch.inference_mode():
        predicted = F.normalize(writer(features), dim=-1)
        expected = F.normalize(targets, dim=-1)
        similarities = (predicted * expected).sum(-1)
        selected = (predicted @ F.normalize(candidates, dim=-1).T).argmax(-1).cpu().tolist()
    rows = []
    for index, case in enumerate(cases):
        rows.append({**case, 'cosine': float(similarities[index]),
                     'candidate_top1': candidate_values[selected[index]] == case['value']})
    groups = {}
    for names in ('train', 'held'):
        for values in ('train', 'held'):
            subset = [row for row in rows if row['name_split'] == names and row['value_split'] == values]
            groups[f'{names}_name_{values}_value'] = {
                'count': len(subset), 'cosine_mean': sum(row['cosine'] for row in subset) / len(subset),
                'cosine_above_0_9': sum(row['cosine'] >= .9 for row in subset),
                'candidate_top1': sum(row['candidate_top1'] for row in subset)}
    return groups, rows


def make_writer(width, seed, identity):
    torch.manual_seed(seed)
    writer = TextSemanticWriter(width)
    if identity:
        with torch.no_grad():
            torch.nn.init.eye_(writer.projection.weight)
    return writer


def main(output=OUT, identity=False):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA:
        raise ValueError('Base weight digest mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    pool = prior.stable_cjk_pool(tokenizer)
    random.Random(prior.SEED).shuffle(pool)
    train = pool[:prior.TRAIN_COUNT]
    held = pool[prior.TRAIN_COUNT:prior.TRAIN_COUNT + prior.HELD_COUNT]
    if len(held) != prior.HELD_COUNT:
        raise ValueError('Insufficient stable value tokens')
    cases = create_cases(prior.TRAIN_NAMES, prior.HELD_NAMES, train, held)
    model = load_full_streamed().eval()
    features, targets = collect(model, tokenizer, cases)
    values = train + held
    token_ids = [tokenizer.encode(value, add_special_tokens=False)[0] for value in values]
    candidates = model.model.embed_tokens(torch.tensor(token_ids, device='cuda')).detach().float()
    del model
    torch.cuda.empty_cache()
    targets = targets.to('cuda')
    training = torch.tensor([i for i, case in enumerate(cases)
                             if case['name_split'] == 'train' and case['value_split'] == 'train'], device='cuda')
    results, weights = [], {}
    for seed in SEEDS:
        for layer, cpu_features in features.items():
            writer = make_writer(targets.shape[-1], seed, identity).to('cuda')
            source = cpu_features.to('cuda')
            before, _ = score(writer, source, targets, cases, values, candidates)
            optimizer = torch.optim.AdamW(writer.parameters(), lr=prior.LR, weight_decay=1e-5, foreach=False)
            train_features, train_targets = source.index_select(0, training), targets.index_select(0, training)
            trace = []
            for step in range(prior.STEPS):
                prediction = writer(train_features)
                cosine = 1 - F.cosine_similarity(prediction, train_targets, dim=-1).mean()
                scale = train_targets.norm(dim=-1).mean().clamp_min(1e-6)
                loss = cosine + .25 * ((prediction - train_targets) / scale).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(writer.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if step in (0, prior.STEPS - 1):
                    trace.append({'step': step + 1, 'loss': float(loss.detach())})
            summary, rows = score(writer, source, targets, cases, values, candidates)
            results.append({'seed': seed, 'layer': layer, 'before': before,
                            'summary': summary, 'rows': rows, 'trace': trace})
            weights[f'{seed}:{layer}'] = {k: v.detach().cpu() for k, v in writer.state_dict().items()}
            print(json.dumps({'seed': seed, 'layer': layer, 'summary': summary}, ensure_ascii=False), flush=True)
            del writer, optimizer, source
    if base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA:
        raise ValueError('Base weights changed')
    output.mkdir()
    path = output / 'writers.pt'
    atomic_save({'format': 'spora-writer-layer-control-v1', 'base_sha256': base.WEIGHT_SHA,
                 'weights': weights, 'train_values': train, 'held_values': held}, path)
    report = {'base_sha256': base.WEIGHT_SHA, 'weights_sha256': base.sha(path),
              'split_sha256': hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode('utf-8')).hexdigest(),
              'steps': prior.STEPS, 'seeds': SEEDS, 'identity_initialization': identity,
              'results': results, 'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Oracle single-token value span; synthetic fixed drink sentence.',
                              'Restricted 320-token candidate ranking, not full vocabulary or generation.',
                              'Frozen unmodified base, no SPORA replacements or persistent memory.',
                              'Same token pool as prior development experiment, not sealed evaluation.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--identity-init', action='store_true')
    args = parser.parse_args()
    main(args.output, args.identity_init)
