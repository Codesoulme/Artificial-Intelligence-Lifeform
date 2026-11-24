"""Measure what remains readable when lexical fast weights stop growing."""

import argparse
import json
from pathlib import Path
import struct
from time import perf_counter

import torch
from torch.nn import functional as F

from scripts.audit_spora_fast_weight_lexical_write import base, compress_dimension, SOURCE
from scripts.deepseek_spora_full_stream_loader import stream_fp16_tensor
from spora.research_ttt_ntp import LowRankFastWeight


RANKS = (1, 2, 4)


def vocabulary(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_size <= min(100_000_000, size - 8):
            raise ValueError('Invalid safetensors header length')
        weights = json.loads(stream.read(header_size))
        embedding = stream_fp16_tensor(stream, weights['model.embed_tokens.weight'], 8 + header_size, size, 'cpu')
    return F.normalize(embedding.index_select(1, compress_dimension(embedding.shape[1])).float(), dim=-1)


def bounded_replay(keys, targets, rank, *, eta, ridge):
    matrix = keys.new_zeros(targets.shape[1], keys.shape[1])
    for key, target in zip(keys, targets, strict=True):
        update = eta * (target - matrix @ key) / (key.square().sum() + ridge)
        matrix += torch.outer(update, key)
        left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
        matrix = (left[:, :rank] * singular[:rank]) @ right[:rank]
    return keys @ matrix.T


def decode(vectors, codebook):
    return (F.normalize(vectors, dim=-1) @ codebook.T).argmax(-1).tolist()


def evaluate(payload, reference, codebook, *, eta, ridge):
    records = {}
    for name, (keys, factors) in payload['factors'].items():
        state = LowRankFastWeight(keys, factors)
        targets = state.recover_targets(len(keys), eta=eta, ridge=ridge)
        lengths = payload['event_lengths'][name]
        expected = [row['expected_ids'] for row in reference[name]]
        if lengths != [len(event) for event in expected] or sum(lengths) != len(keys):
            raise ValueError('Saved event boundaries do not match the lexical audit')
        modes = {'ordered_factors': targets, 'full_matrix': state.read(keys)}
        modes.update({f'rank_{rank}': bounded_replay(keys, targets, rank, eta=eta, ridge=ridge)
                      for rank in RANKS})
        modes['separate_version_rank_4'] = torch.cat([
            bounded_replay(event_keys, event_targets, min(4, len(event_keys)), eta=eta, ridge=ridge)
            for event_keys, event_targets in zip(keys.split(lengths), targets.split(lengths), strict=True)
        ])
        reads = {}
        for mode, vectors in modes.items():
            flat = decode(vectors, codebook)
            reads[mode] = [chunk.tolist() for chunk in torch.tensor(flat).split(lengths)]
        if reads['ordered_factors'] != [row['decoded_ids'] for row in reference[name]]:
            raise ValueError('CPU reference read differs from the saved GPU audit')
        records[name] = {'expected': expected, 'reads': reads}
    return records


def summarize(records):
    result = {}
    for mode in next(iter(records.values()))['reads']:
        latest, corrected, historical = [], [], []
        for row in records.values():
            matches = [actual == expected for actual, expected in zip(row['reads'][mode], row['expected'], strict=True)]
            latest.append(matches[-1])
            if len(matches) > 1:
                corrected.append(matches[-1])
                historical.extend(matches[:-1])
        result[mode] = {label: [sum(group), len(group)] for label, group in
                        (('latest', latest), ('corrected', corrected), ('historical', historical))}
    return result


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    weights = base.MODEL / 'model.safetensors'
    adapter_path = SOURCE / 'adapters.pt'
    state_path, reference_path = args.saved / 'states.pt', args.saved / 'result.json'
    paths = (weights, adapter_path, state_path, reference_path)
    hashes = {str(path): base.sha(path) for path in paths}
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if (hashes[str(weights)] != base.WEIGHT_SHA or hashes[str(state_path)] != reference['state_sha256']
            or hashes[str(adapter_path)] != reference['inputs'][str(adapter_path)]):
        raise ValueError('Frozen lexical memory sources do not match the previous audit')
    settings = torch.load(adapter_path, weights_only=True, map_location='cpu')
    payload = torch.load(state_path, weights_only=True, map_location='cpu')
    codebook = vocabulary(weights)
    print('Loaded frozen lexical codebook and saved factors', flush=True)
    records = evaluate(payload, reference['records'], codebook, eta=settings['eta'], ridge=settings['ridge'])
    summary = summarize(records)
    token_count = sum(sum(lengths) for lengths in payload['event_lengths'].values())
    first_keys, first_values = next(iter(payload['factors'].values()))
    factor_pair_bytes = (first_keys.shape[1] + first_values.shape[1]) * first_keys.element_size()
    separate_ranks = sum(min(4, length) for lengths in payload['event_lengths'].values() for length in lengths)
    storage = {'ordered_factor_bytes': reference['factor_bytes'], 'uint32_token_bytes': 4 * token_count,
               'rank_bytes_per_record': {rank: factor_pair_bytes * rank for rank in RANKS},
               'separate_version_factor_bytes': factor_pair_bytes * separate_ranks}
    if any(base.sha(path) != hashes[str(path)] for path in paths):
        raise ValueError('An audit source changed')
    print(json.dumps({'summary': summary, 'storage': storage, 'eta': settings['eta'], 'ridge': settings['ridge']}))
    result = {'sources': hashes, 'summary': summary, 'storage': storage, 'records': records,
              'eta': settings['eta'], 'ridge': settings['ridge'], 'seconds': perf_counter() - started}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved', type=Path)
    parser.add_argument('output', type=Path)
    main(parser.parse_args())
