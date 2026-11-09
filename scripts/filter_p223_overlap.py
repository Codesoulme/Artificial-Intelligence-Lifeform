"""Exclude mixture records with fixed historical full or 20-character matches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.audit_authored_text_overlap import EXPECTED


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(candidate_dir, audit_path, output):
    candidate_dir, audit_path, output = map(Path, (candidate_dir, audit_path, output))
    if output.exists():
        raise FileExistsError(output)
    manifest_raw = (candidate_dir/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    source = (candidate_dir/'candidate.jsonl').read_bytes()
    audit_raw = audit_path.read_bytes()
    audit = json.loads(audit_raw)
    if (sha(source) != manifest['candidate_sha256'] or sha(source) != audit['candidate_sha256']
            or audit.get('status') != 'training_side_overlap_diagnostic'
            or set(audit.get('comparisons', {})) != {'p28', 'v5'}):
        raise ValueError('Mixture or audit identity mismatch')
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    ids = {row['source_original_sha256'] for row in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate mixture identity')
    for name, source_audit in audit['comparisons'].items():
        if source_audit.get('sha256') != EXPECTED[name] or set(source_audit['by_candidate']) != ids:
            raise ValueError('Audit does not cover exact mixture')
    kept, removed = [], []
    for row in rows:
        hits = [name for name, item in audit['comparisons'].items()
                if item['by_candidate'][row['source_original_sha256']]['documents_with_full_containment']
                or item['by_candidate'][row['source_original_sha256']]['windows']['20']['candidate_window_positions']]
        if hits:
            removed.append({'source_original_sha256': row['source_original_sha256'],
                            'origin_source': row['origin_source'], 'matched_sources': hits})
        else:
            kept.append({**row, 'source_row_1based': len(kept)+1})
    if not kept:
        raise ValueError('All mixture records excluded')
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in kept).encode('utf8')
    result = {'status': 'bounded_pilot_candidate_filtered', 'candidate_sha256': sha(raw),
              'source_candidate_sha256': sha(source), 'source_manifest_sha256': sha(manifest_raw),
              'audit_sha256': sha(audit_raw), 'tokenizer_sha256': manifest['tokenizer_sha256'],
              'counts': {'records': len(kept), 'content_tokens': sum(r['content_tokens'] for r in kept),
                         'by_source': dict(Counter(r['origin_source'] for r in kept)), 'removed': len(removed)},
              'removed': removed, 'test_split_read': False,
              'limitations': ['Only exact full and 20-character matches in two historical training files.',
                              'Other histories, paraphrases, source facts and rights remain unverified.',
                              'Model-reviewed sample, not human-certified general Chinese data.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(raw)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'audit', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.candidate, args.audit, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
