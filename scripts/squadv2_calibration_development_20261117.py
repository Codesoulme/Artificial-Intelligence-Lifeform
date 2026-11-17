"""Select article-disjoint development calibration examples without exporting text."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import pyarrow.parquet as parquet
from scripts.audit_chinese_squadv2_20261117 import SOURCE, SHA
from scripts.select_chinese_squadv2_who_20261117 import select

OUT = ROOT / 'runs/squadv2-calibration-development-20261117.json'


def select_calibration(rows, per_group=10):
    evaluation, _ = select(rows)
    lookup = {hashlib.sha256(row['id'].encode('utf8')).hexdigest(): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate identifiers')
    excluded_titles = {lookup[r['digest']]['title_en'] for r in evaluation}
    candidates, _ = select([row for row in rows if row['title_en'] not in excluded_titles], per_group=30)
    chosen, used_titles = [], set()
    for kind in ('known', 'unknown'):
        count = 0
        for item in candidates:
            title = lookup[item['digest']]['title_en']
            if item['kind'] != kind or (kind, title) in used_titles:
                continue
            used_titles.add((kind, title))
            chosen.append({**item, 'title_digest': hashlib.sha256(title.encode('utf8')).hexdigest()})
            count += 1
            if count == per_group:
                break
        if count != per_group:
            raise ValueError('Insufficient disjoint calibration articles')
    return chosen


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SHA:
        raise ValueError('Source mismatch')
    rows = select_calibration(parquet.read_table(SOURCE).to_pylist())
    result = {'selected': len(rows), 'source_sha256': SHA,
              'rows': [{k: r[k] for k in ('digest', 'context_digest', 'title_digest', 'kind')} for r in rows],
              'limitations': ['Machine-translated labels not independently verified.',
                  'Exact English title separation does not establish semantic or pretraining independence.',
                  'Calibration and earlier evaluation belong to the same public validation distribution.',
                  'No real identity, memory or natural dialogue labels.']}
    with OUT.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'selected': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
