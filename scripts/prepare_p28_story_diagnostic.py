"""Build a bounded, traceable story candidate; never starts training."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path

TRAIN_SHA = 'af994a6be4540e8bdef51c446b7a4a434113d187e054f673d8a5251db23e0c90'
DEV_SHA = '3c33ef6bfc01c5d503a029c91e04505241f6ecd425c3084f7c5b35d372972089'
SALT = 'p199-story-subset-v1:'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample(path, split, count):
    def candidates():
        with Path(path).open(encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('split') != split or not isinstance(row.get('text'), str):
                    raise ValueError('Unexpected source record')
                if hashlib.sha256(row['text'].encode('utf8')).hexdigest() != row.get('source_original_sha256'):
                    raise ValueError('Story does not match source hash')
                key = hashlib.sha256((SALT + split + ':' + str(row['source_row_index'])).encode()).hexdigest()
                yield key, row['source_row_index'], row
    chosen = heapq.nsmallest(count, candidates())
    if len(chosen) != count:
        raise ValueError('Insufficient source records')
    return [row for _, _, row in chosen]


def prepare(train, development, output, train_count=512, dev_count=32):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha(train) != TRAIN_SHA or sha(development) != DEV_SHA:
        raise ValueError('Source digest mismatch')
    if not 1 <= train_count <= 4096 or not 1 <= dev_count <= 256:
        raise ValueError('Candidate size exceeds diagnostic bounds')
    splits = {'train': sample(train, 'train', train_count),
              'development': sample(development, 'development', dev_count)}
    groups = [row['group'] for rows in splits.values() for row in rows]
    originals = [row['source_original_sha256'] for rows in splits.values() for row in rows]
    if len(set(groups)) != len(groups) or len(set(originals)) != len(originals):
        raise ValueError('Selected records share a group or exact source text')
    output.mkdir(parents=False)
    records, provenance = [], []
    for split, rows in splits.items():
        for index, row in enumerate(rows):
            name = f'{split}-{index:04d}.txt'
            raw = row['text'].encode('utf8')
            (output / name).write_bytes(raw)
            records.append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'split': split,
                            'source': 'Gabrui/multilingual_TinyStories@7fb5f327b907318dd95877a366e288f1971b64c1',
                            'license': 'source-declared CDLA-Sharing-1.0; upstream rights unverified'})
            provenance.append({'path': name, 'source_row_index': row['source_row_index'],
                               'source_group': row['group'], 'source_original_sha256': row['source_original_sha256']})
    manifest = {'schema': 'spora-text-v1', 'documents': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    summary = {'status': 'candidate_not_approved_for_training', 'training_started': False,
               'source_train_sha256': TRAIN_SHA, 'source_development_sha256': DEV_SHA,
               'manifest_sha256': sha(output / 'manifest.json'), 'provenance_sha256': sha(output / 'provenance.json'),
               'selection': 'lowest SHA256 of fixed salt, split and source row index',
               'splits': {split: {'documents': len(rows), 'characters': sum(len(row['text']) for row in rows),
                                  'utf8_bytes': sum(len(row['text'].encode('utf8')) for row in rows)}
                          for split, rows in splits.items()},
               'limitations': ['Machine-translated synthetic fiction; review covers only a sample of its parent pool.',
                               'No semantic or historical-corpus overlap proof.',
                               'Upstream rights and downstream obligations require separate review.',
                               'Development was used previously and is not final evaluation.',
                               'No training approval.']}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=512)
    parser.add_argument('--dev-count', type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train, args.development, args.output,
                             train_count=args.train_count, dev_count=args.dev_count), ensure_ascii=False))


if __name__ == '__main__':
    main()
