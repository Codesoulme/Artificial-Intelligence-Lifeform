"""Audit an expanded story curriculum against an existing hybrid interface."""
import argparse
import json
from pathlib import Path

import torch

from spora.hybrid_tokenizer import HybridTokenizer
from spora.text_data import overlap_audit
from spora.train_hybrid import read_non_test_documents, sha


def audit(old_manifest, new_manifest, tokenizer_path, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    old_sha, new_sha, tokenizer_sha = sha(old_manifest), sha(new_manifest), sha(tokenizer_path)
    if checkpoint.get('format') != 'spora-hybrid-training-v1':
        raise ValueError('Expected a hybrid training checkpoint')
    if checkpoint['provenance']['manifest_sha256'] != old_sha or checkpoint['provenance']['vocab_sha256'] != tokenizer_sha:
        raise ValueError('Checkpoint provenance mismatch')
    tokenizer = HybridTokenizer.from_file(tokenizer_path)
    old_docs, new_docs = read_non_test_documents(old_manifest), read_non_test_documents(new_manifest)
    ids = lambda docs, split: {row['sha256'] for row in docs if row['split'] == split}
    old_train, new_train = ids(old_docs, 'train'), ids(new_docs, 'train')
    if not old_train < new_train or ids(old_docs, 'development') != ids(new_docs, 'development'):
        raise ValueError('Require a strict train expansion and unchanged development documents')
    if not set(tokenizer.symbols) <= set(''.join(r['text'] for r in old_docs if r['split'] == 'train')):
        raise ValueError('Tokenizer contains characters absent from previous training data')
    overlap = overlap_audit(new_docs)
    stats = {}
    for split in ('train', 'development'):
        rows = [row for row in new_docs if row['split'] == split]
        stats[split] = {'documents': len(rows), 'characters': sum(len(r['text']) for r in rows),
                        'utf8_bytes': sum(len(r['text'].encode('utf8')) for r in rows),
                        'fallback_characters': sum(sum(c not in tokenizer.ids for c in r['text']) for r in rows),
                        'roundtrip_failures': sum(tokenizer.decode(tokenizer.encode(r['text'])) != r['text'] for r in rows)}
    return {'status': 'transfer_diagnostic_not_training_approval', 'manifest_sha256': new_sha,
            'previous_manifest_sha256': old_sha, 'tokenizer_sha256': tokenizer_sha,
            'source_checkpoint_sha256': sha(checkpoint_path), 'hybrid_vocab': tokenizer.payload(),
            'old_train_documents': len(old_train), 'new_train_documents': len(new_train),
            'additional_train_documents': len(new_train - old_train), 'splits': stats,
            'overlap_train_dev': overlap, 'test_split_read': False,
            'limitations': ['Old training documents are included and may have been seen previously.',
                            'No semantic near-duplicate or historical-corpus isolation proof.',
                            'Development documents are unchanged and not a final evaluation.',
                            'Source is noisy translated synthetic fiction; this audit does not approve training.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-manifest', 'new-manifest', 'tokenizer', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = audit(args.old_manifest, args.new_manifest, args.tokenizer, args.checkpoint)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('old_train_documents', 'new_train_documents',
                                                     'additional_train_documents', 'splits', 'overlap_train_dev')}))


if __name__ == '__main__':
    main()
