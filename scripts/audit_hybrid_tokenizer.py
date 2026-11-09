"""Measure a training-only character vocabulary with lossless byte fallback."""
import argparse
import hashlib
import json
from pathlib import Path

from spora.codepoint import CodepointVocabulary
from spora.hybrid_tokenizer import HybridTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(manifest_path, vocab_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('schema') != 'spora-text-v1':
        raise ValueError('Invalid manifest')
    old_vocab = CodepointVocabulary.from_file(vocab_path)
    tokenizer = HybridTokenizer(old_vocab.symbols)
    split_rows = {'train': [], 'development': []}
    train_chars = set()
    for row in manifest['documents']:
        if row['split'] not in split_rows:
            continue
        path = (manifest_path.parent / row['path']).resolve()
        if not path.is_relative_to(manifest_path.parent) or sha(path) != row['sha256']:
            raise ValueError('Document path or digest mismatch')
        text = path.read_bytes().decode('utf-8-sig')
        if row['split'] == 'train':
            train_chars.update(text)
        split_rows[row['split']].append(text)
    if not all(split_rows.values()):
        raise ValueError('Training and development documents required')
    if set(tokenizer.symbols) != train_chars:
        raise ValueError('Vocabulary does not exactly match the training characters')
    result = {}
    for split, documents in split_rows.items():
        stats = {'documents': len(documents), 'characters': 0, 'utf8_bytes': 0,
                 'fallback_characters': 0, 'hybrid_tokens': 0, 'roundtrip_failures': 0}
        for text in documents:
            tokens = tokenizer.encode(text)
            stats['characters'] += len(text)
            stats['utf8_bytes'] += len(text.encode('utf8'))
            stats['fallback_characters'] += sum(char not in tokenizer.ids for char in text)
            stats['hybrid_tokens'] += len(tokens) - 2
            stats['roundtrip_failures'] += tokenizer.decode(tokens) != text
        result[split] = stats
    return {'status': 'representation_diagnostic_not_training', 'test_split_read': False,
            'manifest_sha256': sha(manifest_path), 'source_vocab_sha256': sha(vocab_path),
            'hybrid_vocab': tokenizer.payload(), 'splits': result,
            'limitations': ['Counts do not establish language ability.',
                            'Vocabulary and measurements are specific to the selected training corpus.',
                            'New tokenizer semantics require a separately bound, freshly trained interface.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.vocab)
    with Path(args.output).open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result['splits'], ensure_ascii=False))


if __name__ == '__main__':
    main()
