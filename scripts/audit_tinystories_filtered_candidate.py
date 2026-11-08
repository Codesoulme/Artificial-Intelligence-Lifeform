"""Audit the P166 TinyStories candidate against prior local story artifacts.

This is a mechanical contamination and partition audit only. It never trains,
never changes the candidate, and never treats the result as training approval.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = ROOT / 'runs/p166-tinystories-filtered-candidate-20261109'
DEFAULT_LEGACY = {
    'train': ROOT / 'runs/p28-stories-data/candidate-v3/pretrain-train.jsonl',
    'development': ROOT / 'runs/p28-stories-data/candidate-v3/story-development.jsonl',
    'test': ROOT / 'runs/p28-stories-data/candidate-v3/story-test.jsonl',
}
SPLIT_SALT = 'spora-tinystories-p168-diagnostic-v1:'


def sha256_path(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    """Normalize finite lexical comparison text without changing source files."""
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text).casefold())


def row_text(row):
    for key in ('text', 'story', 'content'):
        value = row.get(key)
        if isinstance(value, str):
            return value
    raise ValueError('legacy row has no text field')


def read_candidate(directory, expected_records=57):
    manifest_path = directory / 'manifest.json'
    candidate_path = directory / 'candidate.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('status') != 'candidate_not_approved_for_training':
        raise ValueError('candidate status is not the expected non-approval state')
    if manifest.get('training_approval') is not False or manifest.get('training_started') is not False:
        raise ValueError('candidate manifest claims training approval or execution')
    if manifest.get('candidate_sha256') != sha256_path(candidate_path):
        raise ValueError('candidate hash does not match manifest')
    rows = [json.loads(line) for line in candidate_path.read_text(encoding='utf8').splitlines()]
    if len(rows) != manifest.get('records') or (expected_records is not None and len(rows) != expected_records):
        raise ValueError('unexpected candidate record count')
    ids = [row.get('record_id') for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('candidate record ids are not unique')
    return manifest, rows


def duplicate_summary(values):
    groups = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    repeated = {key: indexes for key, indexes in groups.items() if len(indexes) > 1}
    return {
        'unique_values': len(groups),
        'documents_in_repeated_groups': sum(len(v) for v in repeated.values()),
        'duplicate_occurrences_beyond_first': sum(len(v) - 1 for v in repeated.values()),
        'largest_group': max((len(v) for v in groups.values()), default=0),
        'group_count': len(repeated),
    }


def candidate_internal(rows):
    normalized = [normalize(row_text(row)) for row in rows]
    result = {
        'exact_text': duplicate_summary(normalized),
        'prefix_96': duplicate_summary([text[:96] for text in normalized if len(text) >= 96]),
        'source_row_indices_unique': len({row.get('source_row_index_0based') for row in rows}) == len(rows),
        'story_family_metadata': 'unavailable in source candidate; no family grouping claim made',
    }
    return result


def diagnostic_split(record_id):
    digest = hashlib.sha256((SPLIT_SALT + record_id).encode('utf8')).hexdigest()
    bucket = int(digest, 16) % 10
    return ('test' if bucket == 0 else 'development' if bucket == 1 else 'train'), digest


def prior_overlap(rows, legacy_paths):
    candidates = []
    for row in rows:
        text = normalize(row_text(row))
        candidates.append({
            'record_id': row['record_id'],
            'normalized': text,
            'digest': hashlib.sha256(text.encode('utf8')).hexdigest(),
            'prefix_96': text[:96] if len(text) >= 96 else None,
        })
    exact = {item['digest']: [] for item in candidates}
    prefixes = {item['prefix_96']: [] for item in candidates if item['prefix_96']}
    matches = defaultdict(lambda: {'exact_full': [], 'leading_96_contained': []})
    scanned = {}
    for split, path in legacy_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        count = 0
        with path.open(encoding='utf8') as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                old = normalize(row_text(row))
                old_digest = hashlib.sha256(old.encode('utf8')).hexdigest()
                for item in candidates:
                    if old_digest == item['digest'] and old == item['normalized']:
                        matches[item['record_id']]['exact_full'].append({'split': split, 'line': line_number})
                    if item['prefix_96'] and item['prefix_96'] in old:
                        matches[item['record_id']]['leading_96_contained'].append({'split': split, 'line': line_number})
                count += 1
        try:
            display_path = str(path.relative_to(ROOT))
        except ValueError:
            display_path = str(path)
        scanned[split] = {'path': display_path, 'sha256': sha256_path(path), 'records': count}
    for item in candidates:
        value = matches[item['record_id']]
        value['exact_full'] = sorted(value['exact_full'], key=lambda x: (x['split'], x['line']))
        value['leading_96_contained'] = sorted(value['leading_96_contained'], key=lambda x: (x['split'], x['line']))
    return {
        'legacy_inputs': scanned,
        'records_with_exact_full_overlap': sum(bool(v['exact_full']) for v in matches.values()),
        'records_with_leading_96_containment': sum(bool(v['leading_96_contained']) for v in matches.values()),
        'records_with_any_declared_overlap': sum(bool(v['exact_full'] or v['leading_96_contained']) for v in matches.values()),
        'by_candidate': dict(matches),
        'method': 'NFKC + casefold + whitespace removal; exact normalized equality and candidate leading 96 characters contained anywhere in each prior story record',
        'limitations': ['Finite lexical audit only; shifted, paraphrased, translated and semantic overlap can remain.', 'The prior artifacts are the local P28 story files, not every ancestor corpus.', 'No source story-family metadata was available, so family isolation is unresolved.'],
    }


def audit(candidate_dir, legacy_paths):
    manifest, rows = read_candidate(candidate_dir, expected_records=None)
    split_counts = Counter()
    split_digests = {}
    for row in rows:
        split, digest = diagnostic_split(row['record_id'])
        split_counts[split] += 1
        split_digests[row['record_id']] = digest
    overlap = prior_overlap(rows, legacy_paths)
    return {
        'status': 'candidate_not_approved_for_training',
        'training_approval': False,
        'candidate_manifest_sha256': sha256_path(candidate_dir / 'manifest.json'),
        'candidate_sha256': manifest['candidate_sha256'],
        'candidate_records': len(rows),
        'candidate_internal': candidate_internal(rows),
        'diagnostic_partition': {
            'salt': SPLIT_SALT,
            'rule': 'sha256(UTF8(salt + record_id)) interpreted big-endian modulo 10: 0=test, 1=development, otherwise train',
            'counts': dict(sorted(split_counts.items())),
            'record_digests': split_digests,
            'purpose': 'planning only; not an independent final evaluation set and not a training authorization',
        },
        'prior_overlap': overlap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.candidate, DEFAULT_LEGACY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({
        'output': str(args.output),
        'candidate_records': result['candidate_records'],
        'prior_overlap': {
            key: result['prior_overlap'][key]
            for key in ('records_with_exact_full_overlap', 'records_with_leading_96_containment', 'records_with_any_declared_overlap')
        },
        'diagnostic_partition': result['diagnostic_partition']['counts'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
