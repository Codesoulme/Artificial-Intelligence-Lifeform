"""Build a mechanically isolated TinyStories candidate; never trains."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_tinystories_filtered_candidate import (
    DEFAULT_CANDIDATE,
    SPLIT_SALT,
    diagnostic_split,
    read_candidate,
    sha256_path,
)

DEFAULT_AUDIT = ROOT / 'runs/p168-tinystories-isolation-audit-20261109/audit.json'


def build(candidate_dir, audit_path, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest, rows = read_candidate(candidate_dir, expected_records=None)
    audit = json.loads(audit_path.read_text(encoding='utf8'))
    if audit.get('candidate_sha256') != manifest.get('candidate_sha256'):
        raise ValueError('audit and candidate hashes do not match')
    if audit.get('status') != 'candidate_not_approved_for_training' or audit.get('training_approval') is not False:
        raise ValueError('audit is not a non-approval audit')
    by_candidate = audit.get('prior_overlap', {}).get('by_candidate', {})
    excluded = {
        row['record_id'] for row in rows
        if by_candidate.get(row['record_id'], {}).get('exact_full')
        or by_candidate.get(row['record_id'], {}).get('leading_96_contained')
    }
    kept = [row for row in rows if row['record_id'] not in excluded]
    if not kept:
        raise ValueError('isolation removed every candidate record')
    output_dir.mkdir(parents=False)
    candidate_path = output_dir / 'candidate.jsonl'
    with candidate_path.open('x', encoding='utf8', newline='\n') as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    partition = Counter(diagnostic_split(row['record_id'])[0] for row in kept)
    result = {
        'status': 'candidate_not_approved_for_training',
        'training_approval': False,
        'training_started': False,
        'repository': manifest['repository'],
        'revision': manifest['revision'],
        'config': manifest['config'],
        'split': manifest['split'],
        'source_candidate_manifest_sha256': sha256_path(candidate_dir / 'manifest.json'),
        'source_candidate_sha256': manifest['candidate_sha256'],
        'isolation_audit_sha256': sha256_path(audit_path),
        'source_shard_sha256': manifest['source_shard_sha256'],
        'tokenizer_sha256': manifest['tokenizer_sha256'],
        'records': len(kept),
        'excluded_prior_story_overlap_records': len(excluded),
        'excluded_record_ids': sorted(excluded),
        'candidate_sha256': sha256_path(candidate_path),
        'diagnostic_partition': {
            'salt': SPLIT_SALT,
            'rule': 'same P168 diagnostic rule; planning only, not final evaluation or training authorization',
            'counts': dict(sorted(partition.items())),
        },
        'license_declared': manifest['license_declared'],
        'attribution_required': manifest['attribution_required'],
        'limitations': [
            'Only 50 stories remain after exact overlap removal; this is far too small for Chinese pretraining or chat.',
            'The source review was a single development-agent review, not independent human or blind review.',
            'No source story-family metadata was available; semantic, translated and paraphrased overlap can remain.',
            'Candidate remains explicitly unapproved and must not be used as a release or acceptance set.',
        ],
    }
    with (output_dir / 'manifest.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.candidate, args.audit, args.output)
    print(json.dumps({
        'output': str(args.output),
        'records': result['records'],
        'excluded_prior_story_overlap_records': result['excluded_prior_story_overlap_records'],
        'diagnostic_partition': result['diagnostic_partition']['counts'],
        'training_approval': result['training_approval'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
