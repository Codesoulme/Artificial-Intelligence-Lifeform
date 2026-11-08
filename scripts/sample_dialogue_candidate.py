"""Select a fixed group-level train sample for full local review."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=761109)
    p.add_argument('--count', type=int, default=40)
    args = p.parse_args()
    data, output = ROOT / args.data, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    manifest_path, train = data/'manifest.json', data/'train.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if sha(train) != manifest['files'][train.name]:
        raise ValueError('Training candidate differs from manifest')
    if not 1 <= args.count <= 200:
        raise ValueError('Expected bounded sample')
    groups = collections.defaultdict(list)
    with train.open(encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            groups[row['group']].append(row)
    if len(groups) < args.count:
        raise ValueError('Insufficient groups')
    rng = random.Random(args.seed)
    selected = rng.sample(sorted(groups), args.count)
    sample = [rng.choice(groups[group]) for group in selected]
    result = dict(status='for_full_review_not_approved_for_training', source_sha256=sha(train),
                  manifest_sha256=sha(manifest_path), evaluator_sha256=sha(Path(__file__)),
                  seed=args.seed, sampling='uniform group, then uniform member',
                  group_count=len(groups), sample_size=len(sample), samples=sample)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))


if __name__ == '__main__':
    main()
