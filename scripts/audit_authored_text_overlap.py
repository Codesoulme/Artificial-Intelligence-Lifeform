"""Compare prepared authored text with two fixed historical training sources."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.storybooks_overlap import blank, build, finish, normalize, scan_record

EXPECTED = {'p28': 'af994a6be4540e8bdef51c446b7a4a434113d187e054f673d8a5251db23e0c90',
            'v5': '4ec0b001db65b7ccabfaa064e56ce6dff3bbf0bb0a3890c116f4b99999bae9a1'}


def compare_rows(rows, sources, key):
    normalized = [normalize(r['text']) for r in rows]
    machine = build(normalized)
    comparisons = {}
    if set(sources) != set(EXPECTED):
        raise ValueError('Two fixed training sources required')
    for name, path in sources.items():
        results = [blank() for _ in rows]
        digest = hashlib.sha256()
        count = 0
        with Path(path).open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                text = json.loads(line).get('text')
                if not isinstance(text, str):
                    raise ValueError('Invalid training record')
                scan_record(machine, normalize(text), results)
                count += 1
        if digest.hexdigest() != EXPECTED[name]:
            raise ValueError('Historical training source mismatch: ' + name)
        comparisons[name] = {'sha256': digest.hexdigest(), 'records': count,
                             'by_candidate': dict(zip([r[key] for r in rows], finish(results)))}
    return comparisons


def audit(candidate_dir, sources):
    root = Path(candidate_dir)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    raw = (root/'candidate.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get('output_sha256'):
        raise ValueError('Candidate digest mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != manifest.get('documents') or {r['work_group'] for r in rows} != {'meow-v2-one-work'}:
        raise ValueError('Candidate grouping mismatch')
    comparisons = compare_rows(rows, sources, 'id')
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
