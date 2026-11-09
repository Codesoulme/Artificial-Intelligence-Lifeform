"""Build a bounded, document-indexed hybrid-token candidate; never trains."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from spora.hybrid_tokenizer import HybridTokenizer

P28_SHA = 'af994a6be4540e8bdef51c446b7a4a434113d187e054f673d8a5251db23e0c90'


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, tokenizer_path, output, *, expected_source_sha, expected_rows):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha_file(source) != expected_source_sha:
        raise ValueError('Training source digest mismatch')
    tokenizer = HybridTokenizer.from_file(tokenizer_path)
    if tokenizer.size > 65536:
        raise ValueError('Vocabulary exceeds uint16 range')
    if type(expected_rows) is not int or expected_rows < 1:
        raise ValueError('Invalid expected row count')
    output.mkdir(parents=False)
    token_hash, index_hash = hashlib.sha256(), hashlib.sha256()
    count = token_count = chars = fallbacks = 0
    with source.open('rb') as src, (output / 'train.u16').open('xb') as tokens_out, \
            (output / 'index.jsonl').open('xb') as index_out:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get('text')
            if row.get('split') != 'train' or not isinstance(text, str) or not text:
                raise ValueError('Unexpected or empty training record')
            if hashlib.sha256(text.encode('utf8')).hexdigest() != row.get('source_original_sha256'):
                raise ValueError('Source text hash mismatch')
            if not isinstance(row.get('group'), str) or type(row.get('source_row_index')) is not int:
                raise ValueError('Missing source identity')
            ids = tokenizer.encode(text)
            raw = np.asarray(ids, dtype='<u2').tobytes()
            index = {'document': count, 'offset': token_count, 'tokens': len(ids),
                     'source_row_index': row['source_row_index'], 'group': row['group'],
                     'text_sha256': row['source_original_sha256']}
            index_bytes = (json.dumps(index, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf8')
            tokens_out.write(raw)
            index_out.write(index_bytes)
            token_hash.update(raw)
            index_hash.update(index_bytes)
            token_count += len(ids)
            count += 1
            chars += len(text)
            fallbacks += sum(char not in tokenizer.ids for char in text)
    if count != expected_rows or sha_file(source) != expected_source_sha:
        raise ValueError('Source count or digest changed during preparation')
    summary = {'status': 'candidate_not_approved_for_training', 'training_started': False,
               'source_sha256': expected_source_sha, 'tokenizer_sha256': sha_file(tokenizer_path),
               'format': 'spora-hybrid-packed-u16le-v1', 'vocab_size': tokenizer.size,
               'documents': count, 'tokens_including_boundaries': token_count,
               'within_document_targets': token_count - count, 'characters': chars,
               'fallback_characters': fallbacks, 'tokens_sha256': token_hash.hexdigest(),
               'index_sha256': index_hash.hexdigest(), 'test_split_read': False,
               'limitations': ['No training approval; existing source sample review does not certify the full corpus.',
                               'Counts do not prove unique text, linguistic quality, rights or semantic isolation.',
                               'The tokenizer is bound to an existing project-trained interface.']}
    (output / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.tokenizer, args.output,
                             expected_source_sha=P28_SHA, expected_rows=100000), ensure_ascii=False))


if __name__ == '__main__':
    main()
