"""Score fixed previously reviewed encyclopedia development passages."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.evaluate_storybooks_conversation import text_nll
from scripts.prepare_p223_mixture import SOURCES, VOCAB_SHA
from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest
from spora.train_hybrid import sha

ROOT = Path(__file__).resolve().parents[1]


def read_development(root):
    root = Path(root)
    rows, hashes = [], {}
    for name, directory, expected in SOURCES:
        if name == 'voyage':
            continue
        directory = root/directory
        manifest_bytes = (directory/'manifest.json').read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != expected:
            raise ValueError('Development manifest changed: '+name)
        manifest = json.loads(manifest_bytes)
        raw = (directory/'development.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['files']['development.jsonl']:
            raise ValueError('Development data changed: '+name)
        hashes[name] = {'manifest_sha256': expected, 'data_sha256': hashlib.sha256(raw).hexdigest()}
        for line in raw.splitlines():
            row = json.loads(line)
            if hashlib.sha256(row['text'].encode('utf8')).hexdigest() != row['text_sha256']:
                raise ValueError('Development text changed')
            rows.append({'source': name, **row})
    if not rows or len({r['text_sha256'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate development passages')
    return rows, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--individual', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--length', type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    parent_sha, tokenizer_sha = sha(args.individual), sha(args.tokenizer)
    if tokenizer_sha != VOCAB_SHA:
        raise ValueError('Tokenizer mismatch')
    checkpoint_hashes = [sha(p) for p in args.checkpoint]
    rows, source_hashes = read_development(ROOT)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos = tokenizer.token_to_id('<|bos|>')
    encoded = []
    for row in rows:
        ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
        if not ids or tokenizer.decode(ids, skip_special_tokens=False) != row['text']:
            raise ValueError('Tokenizer roundtrip mismatch')
        encoded.append(ids)
    torch.set_num_threads(2)
    organism, state, _ = load(args.individual, args.device)
    before = frozen_digest(organism, state)
    results = []
    for path, checkpoint_sha in zip(args.checkpoint, checkpoint_hashes):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = restore_conversation(organism, payload, parent_sha, tokenizer_sha, args.device)
        if not 2 <= args.length <= model.config.context:
            raise ValueError('Invalid context length')
        docs = []
        for row, ids in zip(rows, encoded):
            docs.append({'source': row['source'], 'text_sha256': row['text_sha256'],
                         'utf8_bytes': len(row['text'].encode('utf8')), 'tokens': len(ids),
                         'nll': text_nll(model, state, ids, bos, args.length)})
        total_nll = sum(d['nll'] for d in docs)
        results.append({'checkpoint_sha256': checkpoint_sha, 'documents': docs,
                        'nll_per_token': total_nll/sum(d['tokens'] for d in docs),
                        'bits_per_utf8_byte': total_nll/sum(d['utf8_bytes'] for d in docs)/math.log(2)})
    if (sha(args.individual) != parent_sha or sha(args.tokenizer) != tokenizer_sha
            or any(sha(path) != expected for path, expected in zip(args.checkpoint, checkpoint_hashes))
            or frozen_digest(organism, state) != before):
        raise AssertionError('Protected inputs changed')
    result = {'source_hashes': source_hashes, 'individual_sha256': parent_sha,
              'tokenizer_sha256': tokenizer_sha, 'length': args.length, 'results': results,
              'test_split_read': False,
              'limitations': ['Development passages, not final independent evaluation.',
                              'Canonical token paths, fixed windows and EOS excluded.',
                              'Prior source and semantic overlap cannot be fully ruled out.',
                              'Likelihood is not a dialogue or factual accuracy assessment.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([{'checkpoint_sha256': x['checkpoint_sha256'], 'nll_per_token': x['nll_per_token'],
                       'bits_per_utf8_byte': x['bits_per_utf8_byte']} for x in results]))


if __name__ == '__main__':
    main()
