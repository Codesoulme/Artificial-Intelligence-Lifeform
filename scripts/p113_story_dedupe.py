from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata
from collections import Counter, defaultdict

RUNS = (Path(__file__).resolve().parents[1] / 'runs').resolve()
MASK = (1 << 64) - 1
# Fixed independent affine permutations modulo a prime, with stable BLAKE2 feature hashes.
PRIME = (1 << 61) - 1
COEFF = [(int.from_bytes(hashlib.sha256(f'p113-a-{i}'.encode()).digest()[:8], 'big') % (PRIME - 1) + 1,
          int.from_bytes(hashlib.sha256(f'p113-b-{i}'.encode()).digest()[:8], 'big') % PRIME) for i in range(32)]


def normalized(s):
    return ''.join(unicodedata.normalize('NFKC', s).casefold().split())


def chinese_grams(s):
    # Han unified ideographs, including supplementary-plane extensions.
    chars = ''.join(c for c in unicodedata.normalize('NFKC', s) if
                    '\u3400' <= c <= '\u4dbf' or '\u4e00' <= c <= '\u9fff' or
                    '\U00020000' <= c <= '\U0002fa1f' or '\U00030000' <= c <= '\U000323af')
    return {chars[i:i+5] for i in range(len(chars)-4)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def signature(grams):
    hashes = [int.from_bytes(hashlib.blake2b(g.encode('utf-8'), digest_size=8, person=b'P113gram').digest(), 'big') % PRIME for g in grams]
    return tuple(min((a * h + b) % PRIME for h in hashes) for a, b in COEFF)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')


def run(input_path, output_path, threshold=0.8, max_df=0.1, min_df_cap=100, max_bucket=200, max_pairs=1000000):
    if not 0 < threshold <= 1 or not 0 < max_df <= 1 or min_df_cap < 1 or max_bucket < 2 or max_pairs < 1:
        raise ValueError('invalid threshold or resource limit')
    output_path = Path(output_path).resolve()
    if not output_path.is_relative_to(RUNS) or output_path == RUNS:
        raise ValueError('output must be a new directory beneath runs')
    input_path = Path(input_path).resolve(strict=True)
    if input_path.is_relative_to(output_path):
        raise ValueError('input cannot be inside output')
    # mkdir with exist_ok=False is the exclusive reservation, preventing overwrite.
    output_path.mkdir(parents=False, exist_ok=False)
    rows, grams, source_lines = [], [], []
    raw_hash = hashlib.sha256()
    with input_path.open('rb') as f:
        for line_no, raw in enumerate(f, 1):
            raw_hash.update(raw)
            if not raw.strip():
                raise ValueError(f'blank input line {line_no}')
            row = json.loads(raw)
            if not isinstance(row, dict) or not isinstance(row.get('text'), str) or not isinstance(row.get('group'), str) or not isinstance(row.get('sha256'), str):
                raise ValueError(f'line {line_no}: expected string text/group/sha256')
            digest = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
            # P112 keeps a normalized identity in sha256 and the raw text
            # digest separately; other inputs may use sha256 for raw text.
            raw_digest = row.get('source_original_sha256', row['sha256'])
            if raw_digest != digest:
                raise ValueError(f'line {line_no}: sha256 does not match UTF-8 text')
            if 'source_original_sha256' in row:
                key = re.sub(r'\s+', '', unicodedata.normalize('NFKC', row['text'])).lower()
                if row['sha256'] != hashlib.sha256(key.encode('utf-8')).hexdigest():
                    raise ValueError(f'line {line_no}: normalized identity mismatch')
            rows.append(row)
            source_lines.append(line_no)
    n = len(rows)
    uf = UnionFind(n)
    exact, prefixes, unique = {}, {}, []
    for i, row in enumerate(rows):
        text = row['text']
        if text in exact:
            uf.union(i, exact[text])
        else:
            exact[text] = i
            unique.append(i)
        prefix = normalized(text)[:96]
        if prefix in prefixes:
            uf.union(i, prefixes[prefix])
        else:
            prefixes[prefix] = i
    # Only exact copies are suppressed from indexing; prefix members still take part
    # so a nonrepresentative can bridge two components.
    feature_sets = {i: chinese_grams(rows[i]['text']) for i in unique}
    freq = Counter(g for fs in feature_sets.values() for g in fs)
    cap = max(min_df_cap, math.floor(max_df * len(unique)))
    frequent = {g for g, count in freq.items() if count > cap}
    reasons = defaultdict(set)
    buckets = defaultdict(list)
    for i in unique:
        fs = feature_sets[i]
        if not fs:
            reasons[i].add('fewer_than_five_chinese_characters')
            continue
        if fs & frequent:
            reasons[i].add('frequent_features_removed')
        usable = fs - frequent
        if not usable:
            reasons[i].add('no_indexable_features')
            continue
        sig = signature(usable)
        for band in range(8):
            buckets[(band, sig[band*4:band*4+4])].append(i)
    pairs = set()
    skipped = []
    # Sorting makes budget decisions independent of hash iteration order.
    for key, members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        possible = len(members) * (len(members)-1) // 2
        reason = 'oversized_bucket' if len(members) > max_bucket else ('pair_budget' if len(pairs) + possible > max_pairs else None)
        if reason:
            for i in members:
                reasons[i].add(reason)
            skipped.append({'band': key[0], 'signature': list(key[1]), 'reason': reason,
                            'source_lines': [source_lines[i] for i in members], 'possible_pairs': possible})
            continue
        pairs.update(itertools.combinations(members, 2))
    accepted = 0
    for a, b in sorted(pairs):
        if jaccard(feature_sets[a], feature_sets[b]) >= threshold:
            uf.union(a, b)
            accepted += 1
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    summary_groups = []
    kept = quarantined = 0
    with (output_path / 'candidates.jsonl').open('x', encoding='utf-8', newline='\n') as out, (output_path / 'quarantine.jsonl').open('x', encoding='utf-8', newline='\n') as quarantine:
        for root, members in sorted(groups.items()):
            why = sorted({reason for i in members for reason in reasons[i]})
            representative = min(members)
            state = 'quarantine' if why else 'candidate'
            if why:
                # Retain every original record while review is pending.
                for i in members:
                    quarantine.write(json.dumps(rows[i], ensure_ascii=False) + '\n')
                quarantined += len(members)
            else:
                out.write(json.dumps(rows[representative], ensure_ascii=False) + '\n')
                kept += 1
            summary_groups.append({'representative_line': source_lines[representative], 'member_lines': [source_lines[i] for i in members], 'status': state, 'review_reasons': why})
    write_json(output_path / 'groups.json', summary_groups)
    write_json(output_path / 'unchecked_buckets.json', skipped)
    def file_hash(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    summary = {'status': 'candidate_not_trained', 'input': {'path': str(input_path), 'sha256': raw_hash.hexdigest(), 'records': n},
               'output': {'candidates': kept, 'quarantined_records': quarantined, 'groups': len(groups),
                          'candidates_sha256': file_hash(output_path / 'candidates.jsonl'), 'quarantine_sha256': file_hash(output_path / 'quarantine.jsonl'),
                          'groups_sha256': file_hash(output_path / 'groups.json'), 'unchecked_buckets_sha256': file_hash(output_path / 'unchecked_buckets.json')},
               'parameters': {'threshold': threshold, 'max_df': max_df, 'min_df_cap': min_df_cap, 'effective_df_cap': cap, 'max_bucket': max_bucket, 'max_pairs': max_pairs, 'bands': 8, 'rows_per_band': 4},
               'counts': {'unique_exact_texts': len(unique), 'frequent_features': len(frequent), 'candidate_pairs_verified': len(pairs), 'similar_pairs': accepted, 'unchecked_buckets': len(skipped)},
               'limitations': ['LSH can miss pairs, including pairs above the threshold.', 'Character 5-gram Jaccard is not semantic similarity and can miss paraphrases.', 'A fixed threshold can produce false positives and false negatives; connected components can join endpoints below threshold.', 'Frequent features and unchecked buckets trigger quarantine; other undetected LSH misses remain possible.']}
    write_json(output_path / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Deterministic candidate-only story near-duplicate grouping; no training.')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new directory directly or indirectly under runs; parent must exist')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--max-df', type=float, default=0.1)
    parser.add_argument('--min-df-cap', type=int, default=100)
    parser.add_argument('--max-bucket', type=int, default=200)
    parser.add_argument('--max-pairs', type=int, default=1000000)
    args = parser.parse_args()
    try:
        result = run(args.input, args.output, args.threshold, args.max_df, args.min_df_cap, args.max_bucket, args.max_pairs)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
