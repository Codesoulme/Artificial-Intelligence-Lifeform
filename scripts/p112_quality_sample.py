"""Local P28 content review sample; never produces a training corpus."""
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/p28-stories-data/candidate-v3'
MATERIAL = ROOT / 'runs/p28-stories-material'
OUT = ROOT / 'runs/p112-story-quality-sample'
SEED = 112281109
SIZE = 80
SOURCE_SHA = '3c86b047b0063056f11b17a494c20a79ce8fda2f17bf4d06634a9d332b740302'
MANIFEST_SHA = 'fd57c0ffa3878c1aef3e762d4c4153672c7c899088f6369ceec33663243af5bd'


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identities(text):
    normalized = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
    return sha_text(normalized), sha_text(normalized[:96])


def score(full):
    return sha_text(f'{SEED}:{full}')


def rank(rows, n=SIZE):
    return sorted(rows, key=lambda row: (score(row['full']), row['full']))[:n]


def choose(original, candidates, n=SIZE):
    fulls = {r['full'] for r in original}
    groups = {r['group'] for r in original}
    if len(fulls) != len(original) or len({r['index'] for r in original}) != len(original):
        raise ValueError('Duplicate original identity or source row')
    eligible = [r for r in candidates if r['full'] not in fulls and r['group'] not in groups]
    a, b = rank(original, n), rank(eligible, n)
    if len(a) != n or len(b) != n:
        raise ValueError('Insufficient eligible records')
    return a, b, len(eligible)


def read_original(path):
    rows = []
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            r = json.loads(line)
            rows.append({'full': r['sha256'], 'group': r['group'],
                         'index': r['source_row_index'], 'original': r['source_original_sha256'],
                         'train_line_1based': line_number})
    return rows


def read_candidates(path):
    if any(Path(str(path) + suffix).exists() for suffix in ('-wal', '-shm', '-journal')):
        raise ValueError('Database has sidecars')
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        cursor = db.execute("SELECT full_hash, group_hash, row_index, original_sha, content_tokens "
                            "FROM candidates WHERE split='train' ORDER BY row_index")
        h = hashlib.sha256()
        rows = []
        for full, group, index, original, tokens in cursor:
            h.update((json.dumps([full, group, index, original, tokens], separators=(',', ':')) + '\n').encode('utf-8'))
            rows.append({'full': full, 'group': group, 'index': index, 'original': original, 'content_tokens': tokens})
        return rows, h.hexdigest()
    finally:
        db.close()


def extract(parquet, selected):
    """Read only row groups containing selected rows; retain selected texts only."""
    wanted = {r['index']: r for r in selected}
    if len(wanted) != len(selected):
        raise ValueError('Duplicate selected source row')
    result = {}
    offset = 0
    for group_no in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_no).num_rows
        indices = [i for i in wanted if offset <= i < offset + count]
        if indices:
            column = parquet.read_row_group(group_no, columns=['story'], use_threads=False).column(0)
            for index in indices:
                text = column[index - offset].as_py()
                row = wanted[index]
                if not isinstance(text, str) or identities(text) != (row['full'], row['group']) or sha_text(text) != row['original']:
                    raise ValueError(f'Source identity mismatch at row {index}')
                result[index] = text
        offset += count
    if len(result) != len(selected):
        raise ValueError('Missing source rows')
    return result


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_sha(BASE / 'manifest.json') != MANIFEST_SHA:
        raise ValueError('Manifest mismatch')
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    train = BASE / 'pretrain-train.jsonl'
    if file_sha(train) != manifest['files']['pretrain-train.jsonl']:
        raise ValueError('Original train mismatch')
    provenance = json.loads((MATERIAL / 'provenance.json').read_text(encoding='utf-8'))
    expected = {'repository': 'Gabrui/multilingual_TinyStories',
                'revision': '7fb5f327b907318dd95877a366e288f1971b64c1',
                'path': 'chinese/train-00000-of-00005.parquet', 'split': 'train',
                'local_file': 'train-00000-of-00005.parquet', 'sha256': SOURCE_SHA}
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError('Source provenance mismatch')
    source = MATERIAL / expected['local_file']
    if source.stat().st_size != provenance['bytes'] or file_sha(source) != SOURCE_SHA:
        raise ValueError('Source shard mismatch')
    original = read_original(train)
    if len(original) != manifest['stats']['pretrain_train_documents']:
        raise ValueError('Original count mismatch')
    candidates, metadata_sha = read_candidates(BASE / 'selection.sqlite')
    a, b, eligible = choose(original, candidates)
    by_full = {r['full']: r for r in candidates}
    for row in original:
        match = by_full.get(row['full'])
        if not match or any(row[k] != match[k] for k in ('group', 'index', 'original')):
            raise ValueError('Original record missing or inconsistent in selection')
    sys.path.insert(0, str(ROOT / 'runs/p28-source-review/vendor'))
    import pyarrow as pa
    import pyarrow.parquet as pq
    if pa.__version__ != '19.0.1' or (ROOT / 'runs/p28-source-review/vendor').resolve() not in Path(pa.__file__).resolve().parents:
        raise ValueError('Expected isolated pyarrow 19.0.1')
    pa.set_cpu_count(1)
    texts = extract(pq.ParquetFile(source), a + b)
    meta = {'purpose': 'local content review only; not training candidates', 'ready_for_training': False,
            'warning': 'B may be near-duplicate of held or historical material; semantic isolation and content quality are not guaranteed.',
            'seed': SEED, 'per_condition': SIZE,
            'ranking': 'ascending SHA256(UTF-8 decimal seed + colon + full normalized SHA256), then full hash',
            'A': 'original pretrain train records',
            'B': "selection split=train excluding original full hashes and original group hashes; no held or historical near-duplicate screening",
            'original_count': len(original), 'train_selection_count': len(candidates), 'B_eligible_count': eligible,
            'source_row_index': 'zero-based physical parquet row', 'train_line_1based': 'one-based original JSONL line, A only',
            'source': provenance, 'manifest_sha256': MANIFEST_SHA,
            'original_train_sha256': manifest['files']['pretrain-train.jsonl'],
            'train_selection_metadata_sha256': metadata_sha,
            'metadata_digest_encoding': 'compact JSON arrays [full, group, row, original, tokens], UTF-8 plus LF, ordered by row',
            'limitations': 'SQLite metadata digest is an observed snapshot; the P28 manifest does not bind the database. Parquet reads selected row groups into memory.'}
    OUT.mkdir()
    with (OUT / 'samples.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for condition, rows in [('A', a), ('B', b)]:
            for row in rows:
                record = {'condition': condition, 'source_row_index': row['index'],
                          'normalized_sha256': row['full'], 'group_sha256': row['group'],
                          'source_original_sha256': row['original'], 'rank_sha256': score(row['full']),
                          'text': texts[row['index']]}
                if condition == 'A':
                    record['train_line_1based'] = row['train_line_1based']
                else:
                    record['content_tokens'] = row['content_tokens']
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    meta['sample_file_sha256'] = file_sha(OUT / 'samples.jsonl')
    with (OUT / 'manifest.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'A': len(a), 'B': len(b), 'B_eligible': eligible, 'metadata_sha256': metadata_sha, 'sample_sha256': meta['sample_file_sha256']}))


if __name__ == '__main__':
    main()
