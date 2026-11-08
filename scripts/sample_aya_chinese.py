"""Summarize and sample only Chinese training rows from pinned Aya data."""
import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def normalized(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text or '')).casefold()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=731109)
    p.add_argument('--count', type=int, default=40)
    args = p.parse_args()
    source, output = ROOT / args.input, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.count <= 200 or digest(source) != args.sha256:
        raise ValueError('Invalid count or source checksum')
    columns = ['inputs', 'targets', 'language', 'language_code', 'annotation_type']
    rows, groups, pairs = [], set(), set()
    languages, types = collections.Counter(), collections.Counter()
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512, columns=columns):
        for item in batch.to_pylist():
            if item['language_code'] != 'zho':
                continue
            rows.append(item)
            groups.add(normalized(item['inputs']))
            pairs.add((normalized(item['inputs']), normalized(item['targets'])))
            languages[item['language']] += 1
            types[item['annotation_type']] += 1
    if len(rows) < args.count:
        raise ValueError('Insufficient Chinese rows')
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.count))
    result = dict(status='sample_for_review_not_approved_for_training', source_sha256=args.sha256,
                  evaluator_sha256=digest(Path(__file__)), seed=args.seed, chinese_rows=len(rows),
                  unique_normalized_prompts=len(groups), unique_normalized_pairs=len(pairs),
                  language_counts=languages, annotation_types=types, sample_size=args.count,
                  samples=[dict(chinese_row=i, **rows[i]) for i in indices],
                  scope='Original train split only; selection by language label, no user identifiers. Reannotation, quality and cross-source overlap still require review.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
