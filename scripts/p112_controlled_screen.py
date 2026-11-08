"""P112 bounded lexical screening. Local feature extraction only; never train."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts.prepare_p28_stories import (BASE as P28_BASE, arrow_modules, verified_source,
    iter_stories, identities, digest, file_hash, load_old_held, encode_whole, quality_reasons)

BASE = P28_BASE / 'candidate-v3'
AUDIT = ROOT / 'reports/P112-story-metadata-audit.json'
PIN_MANIFEST = 'fd57c0ffa3878c1aef3e762d4c4153672c7c899088f6369ceec33663243af5bd'
PIN_METADATA = '6ccde8e20bbaeb1e4b921809c103f81f6cb8f84915a742a0d9874f4c706310ad'
SEED = 112281109


def compact(row):
    return (json.dumps(row, separators=(',', ':')) + '\n').encode('ascii')


def select_eligible(rows, old_full, old_groups, held_groups):
    counts = Counter()
    eligible = []
    metadata = hashlib.sha256()
    for full, group, index, original, tokens in rows:
        metadata.update(compact([full, group, index, original, tokens]))
        if full in old_full:
            counts['original_exact'] += 1
        elif group in old_groups:
            counts['original_group_only'] += 1
        elif group in held_groups:
            counts['other_partition_group'] += 1
        else:
            eligible.append((digest(f'{SEED}:{full}'), full, group, index, original, tokens))
    return eligible, counts, metadata.hexdigest()


def screen(text, indexes):
    for tag, index in indexes:
        reason = index.reason(text)
        if reason:
            return tag + '_' + reason
    return None


def index_digest(index):
    # Binds all feature sets and posting memberships without exposing individual values.
    h = hashlib.sha256()
    for label, values in ((b'E', sorted(index.exact)), (b'G', sorted(index.groups))):
        for value in values:
            h.update(label + value.encode('ascii') + b'\n')
    for size in index.sizes:
        h.update(b'S' + str(size).encode('ascii') + b'\n')
    for sig, postings in sorted(index.index.items()):
        h.update(b'P' + sig + b':' + ','.join(map(str, postings)).encode('ascii') + b'\n')
    return h.hexdigest()


def prepare(name, limit):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not 1 <= limit <= 200000:
        raise ValueError('Invalid output name or limit')
    out = ROOT / 'runs' / name
    if out.exists():
        raise ValueError('Output already exists')
    if file_hash(BASE/'manifest.json') != PIN_MANIFEST:
        raise ValueError('P28 manifest differs from fixed P112 snapshot')
    manifest = json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    for name, key in (('prepare_p28_stories.py', 'script_sha256'), ('prepare_p28_prose.py', 'helper_sha256')):
        if file_hash(ROOT/'scripts'/name) != manifest[key]:
            raise ValueError('Pinned P28 helper changed')
    script_sha = file_hash(Path(__file__))
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if audit['train_metadata_sha256'] != PIN_METADATA:
        raise ValueError('P112 audit differs from pinned digest')
    names = ['pretrain-train.jsonl', 'pretrain-train.bin', 'tokenizer.json',
             'story-development.jsonl', 'story-test.jsonl']
    hashes = {n: file_hash(BASE/n) for n in names}
    if any(hashes[n] != manifest['files'][n] for n in names):
        raise ValueError('P28 file hash mismatch')
    source, provenance = verified_source()
    provenance_sha = file_hash(P28_BASE.parent/'p28-stories-material'/'provenance.json')
    dbpath = BASE/'selection.sqlite'
    if any(Path(str(dbpath)+s).exists() for s in ('-wal', '-journal', '-shm')):
        raise ValueError('Database has sidecars; immutable snapshot unavailable')
    dbsha = file_hash(dbpath)
    db = sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    db.execute('PRAGMA query_only=ON')
    if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
        raise ValueError('Database integrity check failed')
    from scripts.prepare_p28_prose import HeldIndex
    held, evidence = load_old_held()
    original_index = HeldIndex()
    old_full, old_groups, old_rows = set(), set(), set()
    with (BASE/'pretrain-train.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            full, group = identities(row['text'])
            if (full, group, digest(row['text'])) != (row['sha256'], row['group'], row['source_original_sha256']):
                raise ValueError('Original train identity mismatch')
            original_index.add(row['text'])
            old_full.add(full); old_groups.add(group); old_rows.add(row['source_row_index'])
    if len(old_full) != 100000 or len(old_rows) != 100000:
        raise ValueError('Original train uniqueness mismatch')
    held_count = 0
    held_groups = set()
    for full, group, rownum, raw, text in db.execute("SELECT full_hash,group_hash,row_index,original_sha,held_text FROM candidates WHERE split != 'train' ORDER BY row_index"):
        if not isinstance(text, str) or identities(text) != (full, group) or digest(text) != raw:
            raise ValueError('Held database identity mismatch')
        held.add(text); held_groups.add(group); held_count += 1
    # Include selected held files independently, validate their manifest-bound bytes.
    for n in ('story-development.jsonl', 'story-test.jsonl'):
        with (BASE/n).open(encoding='utf-8') as f:
            for line in f:
                held.add(json.loads(line)['text'])
    indexes = [('held', held), ('original_train', original_index)]
    index_hashes = {tag: index_digest(index) for tag, index in indexes}
    eligible, excluded, meta_sha = select_eligible(db.execute("SELECT full_hash,group_hash,row_index,original_sha,content_tokens FROM candidates WHERE split='train' ORDER BY row_index"), old_full, old_groups, held_groups)
    if meta_sha != PIN_METADATA or len(eligible) != 449433 or excluded['original_exact'] != 100000:
        raise ValueError('Candidate metadata differs from P112 snapshot')
    # Validate every DB row against immutable source, including all held entries.
    records = {r[2]: (r[0], r[1], r[3], r[4]) for r in db.execute('SELECT full_hash,group_hash,row_index,original_sha,content_tokens FROM candidates')}
    eligible_by_row = {r[3]: r for r in eligible}
    accepted = []
    stats = Counter(excluded)
    seen = 0
    pa, pq = arrow_modules()
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow.names != ['story']:
        raise ValueError('Source schema mismatch')
    print(json.dumps({'stage': 'screening', 'candidates': len(eligible), 'held_records': held_count}), flush=True)
    for ordinal, text in iter_stories(parquet):
        rec = records.get(ordinal)
        if rec is None:
            continue
        seen += 1
        if not isinstance(text, str) or (*identities(text), digest(text)) != rec[:3]:
            raise ValueError('Database/source binding mismatch')
        item = eligible_by_row.get(ordinal)
        if item is None:
            continue
        reason = screen(text, indexes)
        if reason:
            stats['excluded_'+reason] += 1
        elif quality_reasons(text):
            stats['excluded_quality'] += 1
        else:
            accepted.append(item)
    if seen != len(records) or parquet.metadata.num_rows != 575898:
        raise ValueError('Source coverage mismatch')
    accepted.sort()
    selected = accepted[:limit]
    selected_by_row = {r[3]: r for r in selected}
    sample_hash = hashlib.sha256()
    for score, full, group, ordinal, raw, tokens in selected:
        sample_hash.update(compact([score, full, ordinal, tokens]))
    if file_hash(dbpath) != dbsha or file_hash(source) != provenance['sha256']:
        raise ValueError('Inputs changed during screening')
    from tokenizers import Tokenizer, __version__ as tokenizer_version
    if tokenizer_version != '0.22.1':
        raise ValueError('Require tokenizers 0.22.1')
    tokenizer = Tokenizer.from_file(str(BASE/'tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    out.mkdir(parents=False, exist_ok=False)
    shutil.copyfile(BASE/'tokenizer.json', out/'tokenizer.json')
    written = 0
    with (out/'candidates.jsonl').open('x', encoding='utf-8', newline='\n') as f:
        for ordinal, text in iter_stories(parquet):
            item = selected_by_row.get(ordinal)
            if item is None:
                continue
            score, full, group, _, raw, tokens = item
            ids, why = encode_whole(tokenizer, text)
            if why or len(ids) != tokens or identities(text) != (full, group) or digest(text) != raw or screen(text, indexes):
                raise ValueError('Final verification failed')
            f.write(json.dumps({'text': text, 'source_row_1based': ordinal+1,
                'sha256': full, 'source_original_sha256': raw, 'group': group,
                'content_tokens': tokens}, ensure_ascii=False, separators=(',', ':'))+'\n')
            written += 1
    if written != len(selected) or file_hash(dbpath) != dbsha or file_hash(source) != provenance['sha256']:
        raise ValueError('Final input or output verification failed')
    if (file_hash(Path(__file__)) != script_sha or
        any(file_hash(BASE/n) != value for n, value in hashes.items()) or
        any(file_hash(Path(k)) != v['sha256'] for k, v in evidence.items())):
        raise ValueError('Screening inputs changed')
    db.close()
    result = {'status': 'candidate_only_not_trained', 'trained': False, 'ready_for_training': False,
        'semantic_isolation_proven': False, 'manual_review_passed': False,
        'seed': SEED, 'limit': limit, 'selection': 'SHA256(seed:normalized_full_hash), ascending; filter then cap; output in source-row order',
        'source_revision': provenance['revision'], 'source_sha256': provenance['sha256'],
        'source_provenance_sha256': provenance_sha, 'p28_manifest_sha256': PIN_MANIFEST,
        'p28_file_sha256': hashes, 'database_sha256_observed': dbsha,
        'database_note': 'Original manifest did not bind this database; all candidate identities were verified against pinned source and train metadata against P112.',
        'train_metadata_sha256': meta_sha, 'held_index_sha256': index_hashes['held'],
        'original_train_index_sha256': index_hashes['original_train'],
        'index_digest_format': 'SHA256 sorted full/group hashes, sizes in insertion order, sorted 8-byte shingles and posting ordinals',
        'eligible_before_screen': len(eligible), 'screen_counts': dict(stats), 'passed_screen': len(accepted),
        'exported': written, 'selected_metadata_sha256': sample_hash.hexdigest(),
        'files': {'candidates.jsonl': file_hash(out/'candidates.jsonl'), 'tokenizer.json': file_hash(out/'tokenizer.json')},
        'script_sha256': file_hash(Path(__file__)), 'test_sha256': file_hash(ROOT/'tests/test_p112_screen.py'),
        'helper_sha256': {n: file_hash(ROOT/'scripts'/n) for n in ('prepare_p28_stories.py', 'prepare_p28_prose.py')},
        'dependencies': {'pyarrow': pa.__version__, 'tokenizers': tokenizer_version, 'python': sys.version.split()[0]},
        'held_input_sha256': {str(Path(k).relative_to(ROOT)): v['sha256'] for k,v in evidence.items()},
        'limits': ['Normalized exact, first 96 characters, stride-16 16-character shingles: at least 8 and 80% containment. Short, shifted and paraphrased matches may escape; semantic isolation is not proven.', 'No guarantee of novelty relative to historical weights, tokenizer fitting, other corpora, or upstream data outside this source.', 'Local corpus is a candidate, not approved for training.']}
    with (out/'manifest.json').open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=True, indent=2); f.write('\n')
    print(json.dumps({'output': str(out.relative_to(ROOT)), 'exported': written, 'passed_screen': len(accepted), 'exclusions': dict(stats), 'trained': False}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--limit', type=int, default=200000)
    args = parser.parse_args()
    try:
        prepare(args.name, args.limit)
    except Exception as exc:
        # Suppress exception payloads: parsers and library errors may contain protected text.
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__}), file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
