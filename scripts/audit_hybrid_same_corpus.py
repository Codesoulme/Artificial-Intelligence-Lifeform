"""Bind an existing interface to a new architecture diagnostic on the same corpus."""
import argparse
import json
from pathlib import Path

import torch

from spora.hybrid_tokenizer import HybridTokenizer
from spora.text_data import overlap_audit
from spora.train_hybrid import read_non_test_documents, sha


def audit(manifest, tokenizer_path, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    manifest_sha, tokenizer_sha = sha(manifest), sha(tokenizer_path)
    if (payload.get('format') != 'spora-hybrid-training-v1'
            or payload.get('provenance', {}).get('manifest_sha256') != manifest_sha
            or payload.get('provenance', {}).get('vocab_sha256') != tokenizer_sha):
        raise ValueError('Checkpoint corpus or vocabulary mismatch')
    tokenizer = HybridTokenizer.from_file(tokenizer_path)
    documents = read_non_test_documents(manifest)
    chars = set(''.join(d['text'] for d in documents if d['split'] == 'train'))
    if not set(tokenizer.symbols) <= chars:
        raise ValueError('Tokenizer characters missing from training data')
    return {'status': 'architecture_diagnostic_not_training_approval', 'manifest_sha256': manifest_sha,
            'previous_manifest_sha256': manifest_sha, 'tokenizer_sha256': tokenizer_sha,
            'source_checkpoint_sha256': sha(checkpoint_path), 'hybrid_vocab': tokenizer.payload(),
            'overlap_train_dev': overlap_audit(documents), 'test_split_read': False,
            'limitations': ['Same corpus and development data used by source checkpoint.',
                            'This audit does not approve training or establish general language ability.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'tokenizer', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.tokenizer, args.checkpoint)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('manifest_sha256', 'source_checkpoint_sha256', 'overlap_train_dev')}))


if __name__ == '__main__':
    main()
