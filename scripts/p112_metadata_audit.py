"""Non-text audit of P28 candidate metadata; never prepares training data."""
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/p28-stories-data/candidate-v3'
REPORT = ROOT / 'reports/P112-story-metadata-audit.json'
SEED = 112281109


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    manifest = json.loads((BASE / 'manifest.json').read_text(encoding='utf-8'))
    names = ['pretrain-train.jsonl', 'pretrain-train.bin', 'tokenizer.json']
    hashes = {name: sha(BASE / name) for name in names}
    for name, value in hashes.items():
        if value != manifest['files'][name]:
            raise ValueError('Manifest mismatch: ' + name)
    old_full, old_groups, old_rows = set(), set(), set()
    with (BASE / 'pretrain-train.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            old_full.add(row['sha256'])
            old_groups.add(row['group'])
            old_rows.add(row['source_row_index'])
    if len(old_rows) != manifest['stats']['pretrain_train_documents'] or len(old_full) != len(old_rows):
        raise ValueError('Original train count or uniqueness mismatch')
    # immutable avoids journals and writes; SELECT deliberately omits held_text.
    uri = (BASE / 'selection.sqlite').resolve().as_uri() + '?mode=ro&immutable=1'
    db = sqlite3.connect(uri, uri=True)
    try:
        db.execute('PRAGMA query_only=ON')
        counts = dict(db.execute('SELECT split, COUNT(*) FROM candidates GROUP BY split'))
        held_groups = {row[0] for row in db.execute("SELECT DISTINCT group_hash FROM candidates WHERE split != 'train'")}
        metadata_hash = hashlib.sha256()
        ranked = []
        excluded = {'original_exact': 0, 'original_group_only': 0, 'other_partition_group': 0}
        found_original = 0
        for full, group, index, original, tokens in db.execute(
                "SELECT full_hash, group_hash, row_index, original_sha, content_tokens "
                "FROM candidates WHERE split='train' ORDER BY row_index"):
            metadata_hash.update((json.dumps([full, group, index, original, tokens], separators=(',', ':')) + '\n').encode('utf-8'))
            if full in old_full:
                found_original += 1
                excluded['original_exact'] += 1
            elif group in old_groups:
                excluded['original_group_only'] += 1
            elif group in held_groups:
                excluded['other_partition_group'] += 1
            else:
                score = hashlib.sha256(f'{SEED}:{full}'.encode('ascii')).hexdigest()
                ranked.append((score, full, index, tokens))
        if found_original != len(old_full):
            raise ValueError('Original train records missing from selection')
        ranked.sort()
        shortlist = ranked[:200000]
        sample_digest = hashlib.sha256()
        for score, full, index, tokens in shortlist:
            sample_digest.update((json.dumps([score, full, index, tokens], separators=(',', ':')) + '\n').encode('ascii'))
        report = {
            'status': 'blocked_metadata_audit_only', 'ready_for_training': False,
            'new_stories_exported': 0, 'actual_new_independent_targets': 0,
            'seed': SEED, 'selection': 'SHA256(ASCII seed:normalized_full_hash), ascending; first 200000 after metadata exclusions',
            'candidate_split_counts': counts, 'original_train_records': len(old_rows),
            'original_train_groups': len(old_groups), 'excluded_train_metadata': excluded,
            'metadata_eligible_before_near_duplicate_checks': len(ranked),
            'shortlist_metadata_count': len(shortlist),
            'shortlist_metadata_sha256': sample_digest.hexdigest(),
            'train_metadata_sha256': metadata_hash.hexdigest(),
            'metadata_encoding': 'UTF-8 compact JSON arrays in stated order, one LF each; train rows ordered by source row, shortlist by score/full/row/tokens',
            'shortlist_content_token_sum_unverified': sum(x[3] for x in shortlist),
            'old_train_within_document_targets_manifest': manifest['story_splits']['train']['within_document_next_token_targets'],
            'verified_file_sha256': hashes,
            'manifest_sha256': sha(BASE / 'manifest.json'),
            'audit_script_sha256': sha(Path(__file__)),
            'source_sha256_previously_recorded_not_rehashed': manifest['source_provenance']['sha256'],
            'limits': [
                'No candidate text, source parquet or held files were opened; original training JSONL was read only to obtain identity metadata.',
                'SQLite queries select metadata only; the underlying database contains held text and cannot be byte-hashed without reading it.',
                'The original manifest does not bind selection.sqlite. Metadata digests record the observed snapshot, not proof of its original integrity.',
                'Existing full-shard pass records legacy lexical screening but does not retain its held index, and resumed-prefix filtering counts are unavailable.',
                'New story held and original train near-duplicates have not been excluded for this shortlist; no isolated semantic comparison has been performed.',
                'A metadata shortlist is neither a safe corpus nor an actual count of distinct training targets. No material was exported or training performed.',
                'Existing review covers 64 original train examples for a bounded pilot only, not this shortlist.',
            ],
        }
    finally:
        db.close()
    with REPORT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('candidate_split_counts', 'excluded_train_metadata', 'metadata_eligible_before_near_duplicate_checks', 'shortlist_metadata_count', 'new_stories_exported')}))


if __name__ == '__main__':
    main()
