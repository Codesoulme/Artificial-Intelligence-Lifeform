"""Produce a bounded sentence-only audit from a pinned transcript TSV."""
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=701109)
    p.add_argument('--count', type=int, default=40)
    args = p.parse_args()
    source, output = ROOT / args.input, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    with source.open('rb') as f:
        source_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    if source_hash != args.sha256:
        raise ValueError('Transcript checksum mismatch')
    if not 1 <= args.count <= 200:
        raise ValueError('Expected a bounded audit sample')
    seen, unique, rows = set(), [], 0
    with source.open(encoding='utf8', newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rows += 1
            sentence = (row.get('sentence') or '').strip()
            key = re.sub(r'\s+', '', unicodedata.normalize('NFKC', sentence)).casefold()
            if key and key not in seen:
                seen.add(key)
                unique.append(sentence)
    if len(unique) < args.count:
        raise ValueError('Not enough unique sentences')
    indices = sorted(random.Random(args.seed).sample(range(len(unique)), args.count))
    result = dict(status='sample_for_review_not_approved_for_training', source_sha256=source_hash,
                  rows=rows, unique_normalized=len(unique), seed=args.seed, sample_size=args.count,
                  samples=[dict(unique_index=i, sentence=unique[i]) for i in indices])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))


if __name__ == '__main__':
    main()
