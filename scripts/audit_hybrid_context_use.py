"""Compare development next-token losses with earlier context shuffled."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from spora.growing import load
from spora.hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer
from spora.language_training import frozen_digest, load_interface
from spora.train_hybrid import read_non_test_documents, sha


def shuffled_prefix(prefix, seed):
    if len(prefix) < 3:
        raise ValueError('Prefix must include BOS and at least two content tokens')
    middle = list(prefix[1:-1])
    random.Random(seed).shuffle(middle)
    return [prefix[0], *middle, prefix[-1]]


def samples(documents, tokenizer, positions):
    result = []
    for doc in documents:
        if doc['split'] != 'development':
            continue
        tokens = tokenizer.encode(doc['text'])
        for pos in positions:
            if pos >= len(tokens) - 1:
                continue
            seed = int.from_bytes(hashlib.sha256((doc['sha256'] + ':' + str(pos)).encode()).digest()[:8], 'big')
            original = tokens[:pos]
            result.append({'path': doc['path'], 'position': pos, 'prefix': original,
                           'shuffled': shuffled_prefix(original, seed), 'target': tokens[pos]})
    if not result:
        raise ValueError('No eligible development targets')
    return result


@torch.no_grad()
def measure(model, state, rows, device):
    results = []
    for position in sorted({row['position'] for row in rows}):
        group = [row for row in rows if row['position'] == position]
        target = torch.tensor([r['target'] for r in group], device=device)
        orig = torch.tensor([r['prefix'] for r in group], device=device)
        shuffled = torch.tensor([r['shuffled'] for r in group], device=device)
        original_loss = F.cross_entropy(model(orig, state)[:, -1], target, reduction='none')
        shuffled_loss = F.cross_entropy(model(shuffled, state)[:, -1], target, reduction='none')
        for row, a, b in zip(group, original_loss.tolist(), shuffled_loss.tolist()):
            results.append({'path': row['path'], 'position': position, 'original_nll': a, 'shuffled_nll': b})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'checkpoint', 'tokenizer', 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    keys = ('individual', 'checkpoint', 'tokenizer', 'manifest')
    hashes = {key: sha(getattr(args, key)) for key in keys}
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    provenance = payload.get('provenance', {})
    if payload.get('format') != 'spora-hybrid-training-v1' or provenance.get('manifest_sha256') != hashes['manifest']:
        raise ValueError('Checkpoint or manifest mismatch')
    if provenance.get('vocab_sha256') != hashes['tokenizer']:
        raise ValueError('Tokenizer mismatch')
    tokenizer = HybridTokenizer.from_file(args.tokenizer)
    docs = read_non_test_documents(args.manifest)
    rows = samples(docs, tokenizer, (16, 32, 64, 96))
    torch.set_num_threads(2)
    organism, state, _ = load(args.individual, args.device)
    before = frozen_digest(organism, state)
    model = HybridLanguageBridge(organism, tokenizer, hashes['tokenizer'], payload['context_layers']).to(args.device).eval()
    if provenance.get('train_backbone_copy') is True:
        model.enable_language_block_copies()
        model.eval()
    load_interface(model, payload, hashes['individual'])
    scored = measure(model, state, rows, args.device)
    if any(sha(getattr(args, key)) != hashes[key] for key in keys) or before != frozen_digest(organism, state):
        raise AssertionError('Protected inputs changed')
    count = len(scored)
    original = sum(row['original_nll'] for row in scored) / count
    shuffled = sum(row['shuffled_nll'] for row in scored) / count
    result = {'split': 'development', 'test_split_read': False, 'input_sha256': hashes,
              'targets': count, 'documents': len({r['path'] for r in scored}),
              'original_nll': original, 'shuffled_nll': shuffled,
              'difference_shuffled_minus_original': shuffled - original,
              'rows': scored, 'parent_and_inputs_unchanged': True,
              'limitations': ['Same target, BOS, last token, token multiset and length; earlier token order is shuffled.',
                              'Shuffling may produce out-of-distribution sequences; this is not semantic comprehension.',
                              'Targets within a story are correlated, and the development set was previously used.',
                              'A negative or small result does not prove the model never uses context.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('targets', 'documents', 'original_nll', 'shuffled_nll', 'difference_shuffled_minus_original')}))


if __name__ == '__main__':
    main()
