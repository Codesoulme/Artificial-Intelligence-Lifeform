"""Build a bounded train-only quality sample, not a training corpus."""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
NOISE = re.compile(r'https?://|www\.|\{\{|\}\}|\[\[|\]\]|-\{[^}]+\}-|\(\s*[,;]?\s*\)|(?:^|\n)\s*[*#]|参考资料|参考文献|參考資料|參考文獻|參考來源|外部链接|参考書目|外部連結|另見|参见|參見|注釋|註釋|\b(?:isbn|doi)\b', re.I)
CONTACT = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)')


def checksum(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def eligible(text):
    text = unicodedata.normalize('NFKC', text or '').strip()
    if not 250 <= len(text) <= 2500:
        return None
    if NOISE.search(text) or CONTACT.search(text):
        return None
    if sum('\u4e00' <= c <= '\u9fff' for c in text) / len(text) < .45:
        return None
    if sum(c.isdigit() for c in text) / len(text) > .10:
        return None
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=711109)
    p.add_argument('--count', type=int, default=40)
    args = p.parse_args()
    source, output = ROOT / args.input, ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= args.count <= 200 or checksum(source) != args.sha256:
        raise ValueError('Invalid sample size or source checksum')
    rng = random.Random(args.seed)
    sample, train_count, eligible_count, row_count = [], 0, 0, 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=256, columns=['id', 'url', 'title', 'text']):
        for row in batch.to_pylist():
            row_count += 1
            if not row['id'] or not row['url']:
                continue
            # Hold out 10% by stable page identifier. Held-out texts are never sampled.
            bucket = int(hashlib.sha256(str(row['id']).encode('utf8')).hexdigest()[:8], 16) % 100
            if bucket >= 90:
                continue
            train_count += 1
            text = eligible(row['text'])
            if text is None:
                continue
            eligible_count += 1
            item = dict(id=row['id'], url=row['url'], title=row['title'], text=text,
                        text_sha256=hashlib.sha256(text.encode('utf8')).hexdigest())
            if len(sample) < args.count:
                sample.append(item)
            else:
                index = rng.randrange(eligible_count)
                if index < args.count:
                    sample[index] = item
    if len(sample) != args.count:
        raise ValueError('Insufficient candidate articles')
    result = dict(status='sample_for_review_not_approved_for_training', source_sha256=args.sha256,
                  evaluator_sha256=checksum(Path(__file__)), seed=args.seed, rows=row_count,
                  train_rows=train_count, candidate_rows=eligible_count, sample_size=len(sample), samples=sample,
                  scope='Single shard; train bucket only. Mechanical filter and sample do not approve or validate all rows.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}))


if __name__ == '__main__':
    main()
