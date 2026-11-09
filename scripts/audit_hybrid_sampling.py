"""Reconstruct window sampling for a single hybrid training stage."""
import argparse
import json
from pathlib import Path

import torch

from spora.hybrid_tokenizer import HybridTokenizer
from spora.curriculum import CoveringSampler
from spora.train_hybrid import read_non_test_documents, sha


def reconstruct(documents, tokenizer, *, length, steps, batch, seed, sampling='random'):
    if not 2 <= length <= 256 or not 1 <= batch <= 16 or not 0 <= steps <= 10000:
        raise ValueError('Invalid sampling parameters')
    windows = []
    for doc in documents:
        if doc['split'] != 'train':
            continue
        token_count = len(tokenizer.encode(doc['text']))
        for start in range(0, token_count - 1, length - 1):
            windows.append((doc['sha256'], min(length - 1, token_count - 1 - start)))
    if not windows:
        raise ValueError('No training windows')
    picks = []
    if sampling == 'random':
        generator = torch.Generator().manual_seed(seed)
        for _ in range(steps):
            picks.extend(torch.randint(len(windows), (batch,), generator=generator).tolist())
        state = generator.get_state()
    elif sampling == 'covering-v1':
        sampler = CoveringSampler({'all': range(len(windows))}, ['all'], seed)
        for _ in range(steps):
            picks.extend(sampler.take(batch).tolist())
        state = sampler.snapshot()
    else:
        raise ValueError('Unsupported sampling method')
    return windows, picks, state


def summarize(windows, picks, previous_ids):
    seen = set(picks)
    documents = {windows[index][0] for index in picks}
    return {'window_draws': len(picks), 'different_windows': len(seen), 'different_documents': len(documents),
            'documents_in_previous_pool': len(documents & previous_ids),
            'documents_not_in_previous_pool': len(documents - previous_ids),
            'target_presentations': sum(windows[index][1] for index in picks),
            'different_window_targets': sum(windows[index][1] for index in seen)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'previous-manifest', 'tokenizer', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    keys = ('manifest', 'previous_manifest', 'tokenizer', 'checkpoint')
    hashes = {key: sha(getattr(args, key)) for key in keys}
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    provenance = checkpoint.get('provenance', {})
    if (checkpoint.get('format') != 'spora-hybrid-training-v1'
            or provenance.get('manifest_sha256') != hashes['manifest']
            or provenance.get('vocab_sha256') != hashes['tokenizer']):
        raise ValueError('Checkpoint inputs do not match')
    docs = read_non_test_documents(args.manifest)
    previous = read_non_test_documents(args.previous_manifest)
    previous_ids = {row['sha256'] for row in previous if row['split'] == 'train'}
    tokenizer = HybridTokenizer.from_file(args.tokenizer)
    sampling = provenance.get('sampling', 'random')
    windows, picks, rng = reconstruct(docs, tokenizer, length=provenance['length'],
                                      steps=checkpoint['steps'], batch=provenance['batch'], seed=provenance['seed'],
                                      sampling=sampling)
    if sampling == 'random':
        matches = torch.equal(rng, checkpoint['sampler_rng'])
    else:
        actual = checkpoint.get('covering_sampler', {})
        matches = (all(rng[key] == actual.get(key) for key in ('signature', 'orders', 'positions', 'draws', 'counts'))
                   and isinstance(actual.get('rng'), torch.Tensor) and torch.equal(rng['rng'], actual['rng']))
    if not matches:
        raise ValueError('Reconstructed sampler state differs from checkpoint')
    summary = summarize(windows, picks, previous_ids)
    if summary['target_presentations'] != checkpoint['trained_tokens']:
        raise ValueError('Reconstructed target count differs from checkpoint')
    if any(sha(getattr(args, key)) != hashes[key] for key in keys):
        raise AssertionError('Input file changed')
    result = {'input_sha256': hashes, 'test_split_read': False, 'stage_steps': checkpoint['steps'], 'sampling': sampling,
              'available_windows': len(windows), 'sampler_state_matches': True, 'coverage': summary,
              'limitations': ['Stage only; prior stages and semantic overlap are not counted.',
                              'Different window targets include EOS and do not measure knowledge or semantic uniqueness.',
                              'Documents absent from the previous pool may overlap with other historical sources.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['coverage']))


if __name__ == '__main__':
    main()
