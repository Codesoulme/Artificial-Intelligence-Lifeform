"""Draw an independent local review sample of candidate train paragraphs."""
import argparse
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
    p.add_argument('--exclude', action='append', default=[])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--count', type=int, default=60)
    args = p.parse_args()
    data, output = ROOT/args.data, ROOT/args.output
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.count <= 200:
        raise ValueError('Expected bounded review sample')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf8'))
    source = data/'train.jsonl'
    source_sha = sha(source)
    if source_sha != manifest['files'][source.name]:
        raise ValueError('Candidate checksum mismatch')
    excluded = set()
    for name in args.exclude:
        prior = json.loads((ROOT/name).read_text(encoding='utf8'))
        excluded.update(row['group'] for row in prior['samples'])
    rows = [json.loads(line) for line in source.read_text(encoding='utf8').splitlines()]
    pool = [i for i, row in enumerate(rows) if row['group'] not in excluded]
    if len(pool) < args.count:
        raise ValueError('Insufficient eligible rows')
    indices = sorted(random.Random(args.seed).sample(pool, args.count))
    result = dict(seed=args.seed, source_sha256=source_sha, excluded_prior_groups=len(excluded),
                  eligible_pool_rows=len(pool), sampled_indices=indices, samples=[rows[i] for i in indices])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'samples'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
