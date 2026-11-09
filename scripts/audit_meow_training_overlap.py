"""Audit one fixed prose candidate against explicitly named training sources."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.storybooks_overlap import blank, build, finish, normalize, scan_record


def audit(candidate_dir, sources):
    root = Path(candidate_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf8'))
    raw = (root / 'prose.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get('candidate_sha256'):
        raise ValueError('Candidate digest mismatch')
    normalized = normalize(raw.decode('utf8'))
    if len(normalized) < 20:
        raise ValueError('Candidate is too short for comparison')
    machine = build([normalized])
    comparisons = {}
    for name, path in sources.items():
        if 'test' in Path(path).name.lower() or 'development' in Path(path).name.lower():
            raise ValueError('Only explicitly named training sources are allowed')
        result = [blank()]
        h = hashlib.sha256()
        count = 0
        with Path(path).open('rb') as stream:
            for line in stream:
                h.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get('text'), str):
                    raise ValueError('Training record missing text')
                scan_record(machine, normalize(row['text']), result)
                count += 1
        comparisons[name] = {'sha256': h.hexdigest(), 'records': count, 'matches': finish(result)[0]}
    return {'status': 'overlap_diagnostic_not_training_approval', 'candidate_sha256': manifest['candidate_sha256'],
            'normalization': 'NFKC, casefold, remove non-alphanumeric characters',
            'normalized_characters': len(normalized), 'comparisons': comparisons, 'test_split_read': False,
            'limitations': ['Checks only supplied training files and exact or 12/20-character overlap.',
                            'Semantic paraphrases, translation and unscanned historical data remain unverified.',
                            'Short common phrases are not necessarily evidence of shared works.',
                            'No training approval.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--p28-training', required=True)
    parser.add_argument('--v5-training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.candidate, {'p28_training': args.p28_training, 'v5_training': args.v5_training})
    with output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
