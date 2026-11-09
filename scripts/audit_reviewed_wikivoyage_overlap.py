"""Compare the reviewed passage sample with fixed historical training text."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_authored_text_overlap import compare_rows


def audit(candidate_dir, sources):
    root = Path(candidate_dir)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    raw = (root/'candidate.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get('output_sha256'):
        raise ValueError('Candidate digest mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != manifest.get('paragraphs') or len({r['text_sha256'] for r in rows}) != len(rows):
        raise ValueError('Candidate count or identity mismatch')
    for row in rows:
        if hashlib.sha256(row['text'].encode('utf8')).hexdigest() != row['text_sha256']:
            raise ValueError('Candidate text hash mismatch')
    comparisons = compare_rows(rows, sources, 'text_sha256')
    return {'status': 'training_side_overlap_diagnostic', 'candidate_sha256': manifest['output_sha256'],
            'normalization': 'NFKC casefold, remove non-alphanumeric characters',
            'comparisons': comparisons, 'test_split_read': False,
            'limitations': ['Only two fixed historical training sources and exact 12/20-character windows.',
                            'Shared phrases are not necessarily duplicated works; semantic overlap can remain.',
                            'No development or final-test text read; result is not training approval.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'p28', 'v5', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.candidate, {'p28': args.p28, 'v5': args.v5})
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'records': {k: v['records'] for k, v in result['comparisons'].items()},
                      'candidate_sha256': result['candidate_sha256']}))


if __name__ == '__main__':
    main()
