"""Verify packed hybrid candidates without opening development or test sets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_hybrid_packed import sha_file
from spora.hybrid_tokenizer import HybridTokenizer
from spora.language_bridge import BOS, EOS, PAD


def verify(directory, tokenizer_path, source_path):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    if manifest.get('format') != 'spora-hybrid-packed-u16le-v1' or manifest.get('training_started') is not False:
        raise ValueError('Unexpected candidate manifest')
    checks = {'tokenizer_sha256': sha_file(tokenizer_path), 'source_sha256': sha_file(source_path),
              'tokens_sha256': sha_file(root / 'train.u16'), 'index_sha256': sha_file(root / 'index.jsonl')}
    if any(manifest.get(key) != value for key, value in checks.items()):
        raise ValueError('Artifact hash mismatch')
    tokenizer = HybridTokenizer.from_file(tokenizer_path)
    if tokenizer.size != manifest['vocab_size']:
        raise ValueError('Vocabulary size mismatch')
    tokens = np.memmap(root / 'train.u16', dtype='<u2', mode='r')
    if len(tokens) != manifest['tokens_including_boundaries']:
        raise ValueError('Token count mismatch')
    count = offset = fallback_bytes = 0
    with (root / 'index.jsonl').open(encoding='utf8') as index_stream, Path(source_path).open(encoding='utf8') as source:
        nonempty_source = (line for line in source if line.strip())
        for index_line, source_line in zip(index_stream, nonempty_source, strict=True):
            entry, row = json.loads(index_line), json.loads(source_line)
            if entry['document'] != count or entry['offset'] != offset or entry['tokens'] < 2:
                raise ValueError('Invalid document offsets')
            end = offset + entry['tokens']
            if end > len(tokens):
                raise ValueError('Document extends past stream')
            piece = tokens[offset:end].astype(np.int64).tolist()
            if (piece[0] != BOS or piece[-1] != EOS or any(t >= tokenizer.size for t in piece)
                    or any(t in (BOS, EOS, PAD) for t in piece[1:-1])):
                raise ValueError('Invalid document tokens')
            text = tokenizer.decode(piece)
            if (row.get('split') != 'train' or text != row.get('text') or piece != tokenizer.encode(text)
                    or hashlib.sha256(text.encode('utf8')).hexdigest() != entry['text_sha256']
                    or entry['text_sha256'] != row.get('source_original_sha256')
                    or entry['source_row_index'] != row.get('source_row_index') or entry['group'] != row.get('group')):
                raise ValueError('Source and packed document differ')
            fallback_bytes += sum(token < 256 for token in piece)
            offset = end
            count += 1
    if count != manifest['documents'] or offset != len(tokens):
        raise ValueError('Document count or stream coverage mismatch')
    return {'status': 'packed_candidate_verified_not_training_approval', 'documents': count,
            'tokens': offset, 'fallback_byte_tokens': fallback_bytes, 'input_sha256': checks,
            'test_split_read': False, 'limitations': ['Byte-level fidelity does not prove source quality, rights or isolation.',
                                                       'Training remains unapproved.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = verify(args.directory, args.tokenizer, args.source)
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
