"""Score reviewed exploratory stories with project conversation checkpoints."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from scripts.evaluate_storybooks_hybrid import reviewed_documents
from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest
from spora.train_hybrid import sha


@torch.no_grad()
def text_nll(model, state, ids, bos, length):
    if len(ids) == 0 or length < 2:
        raise ValueError('Nonempty tokens and context required')
    tokens = [bos] + ids
    total = 0.0
    for start in range(0, len(tokens) - 1, length - 1):
        piece = torch.tensor([tokens[start:start + length]], dtype=torch.long,
                             device=next(model.parameters()).device)
        logits = model(piece, state)[:, :-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               piece[:, 1:].reshape(-1), reduction='sum')
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite evaluation loss')
        total += float(loss)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('individual', 'tokenizer', 'storybooks-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--length', type=int, default=128)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    parent_sha, tokenizer_sha = sha(args.individual), sha(args.tokenizer)
    checkpoint_hashes = [sha(path) for path in args.checkpoint]
    docs, review_hashes = reviewed_documents(args.storybooks_root)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    bos = tokenizer.token_to_id('<|bos|>')
    if bos is None:
        raise ValueError('Tokenizer has no BOS')
    encoded = []
    for doc in docs:
        ids = tokenizer.encode(doc['text'], add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != doc['text']:
            raise ValueError('Tokenizer does not roundtrip original text')
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
        rows = []
        for doc, ids in zip(docs, encoded):
            text = doc['text']
            rows.append({'story': doc['story'], 'source_sha256': doc['source_sha256'],
                         'characters': len(text), 'utf8_bytes': len(text.encode('utf8')),
                         'content_targets': len(ids), 'total_nll': text_nll(model, state, ids, bos, args.length)})
        byte_count = sum(row['utf8_bytes'] for row in rows)
        results.append({'checkpoint_sha256': checkpoint_sha, 'documents': rows,
                        'bits_per_utf8_byte': sum(row['total_nll'] for row in rows) / byte_count / math.log(2),
                        'total_utf8_bytes': byte_count})
    if (sha(args.individual) != parent_sha or sha(args.tokenizer) != tokenizer_sha
            or any(sha(path) != digest for path, digest in zip(args.checkpoint, checkpoint_hashes))
            or frozen_digest(organism, state) != before):
        raise AssertionError('Protected inputs changed')
    result = {'individual_sha256': parent_sha, 'tokenizer_sha256': tokenizer_sha,
              'review_sha256': review_hashes, 'length': args.length, 'story_count': len(docs),
              'results': results, 'test_split_read': False, 'parent_and_inputs_unchanged': True,
              'limitations': ['Previously reviewed exploratory stories, not final evaluation.',
                              'Canonical token paths only; alternative tokenizations are not summed.',
                              'Fixed token windows reset context and exclude EOS; tokenizers cover different text per window.',
                              'Historical and semantic overlap are not exhaustively excluded.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps([{'checkpoint_sha256': row['checkpoint_sha256'],
                       'bits_per_utf8_byte': row['bits_per_utf8_byte']} for row in results]))


if __name__ == '__main__':
    main()
