"""Check the selected mixture against two fixed historical training files."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_authored_text_overlap import compare_rows


def audit(directory, sources):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    raw = (directory/'candidate.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get('candidate_sha256'):
        raise ValueError('Mixture changed')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != manifest.get('counts', {}).get('records') or len({r['source_original_sha256'] for r in rows}) != len(rows):
        raise ValueError('Mixture identity mismatch')
    return {'status': 'training_side_overlap_diagnostic', 'candidate_sha256': manifest['candidate_sha256'],
            'comparisons': compare_rows(rows, sources, 'source_original_sha256'), 'test_split_read': False,
            'limitations': ['Two training files and exact windows only.',
                            'Semantic paraphrases, other ancestors and final-test overlap remain unverified.']}


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
    print(json.dumps({'candidate_sha256': result['candidate_sha256'],
                      'counts': {k: v['records'] for k, v in result['comparisons'].items()}}))


if __name__ == '__main__':
    main()
