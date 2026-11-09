"""Build a stratified review queue for the extracted AlJabr candidate."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'runs/p178-aljabr-latex-candidate-20261109'


def rank(row):
    return hashlib.sha256(('aljabr-p180:' + row['record_id']).encode('utf8')).hexdigest()


def strata(row):
    length = 'short' if row['chars'] < 160 else 'long' if row['chars'] >= 400 else 'medium'
    formula = 'none' if row['formula_placeholders'] == 0 else 'many' if row['formula_placeholders'] >= 8 else 'few'
    return row['source'], length, formula


def select_queue(rows, limit=80):
    if len(rows) < limit:
        raise ValueError('candidate is smaller than requested queue')
    selected = []
    used = set()
    groups = {}
    for row in rows:
        groups.setdefault((row['source'], row['path'].split('.')[0]), []).append(row)
    sources = sorted({row['source'] for row in rows})
    if not sources or limit % len(sources):
        raise ValueError('queue limit must divide source count')
    per_source = limit // len(sources)
    for source in sources:
        source_rows = [row for row in rows if row['source'] == source]
        source_selected = []
        source_used = set()
        source_groups = {key: value for key, value in groups.items() if key[0] == source}
        for key in sorted(source_groups):
            bucket = sorted(source_groups[key], key=rank)
            source_selected.append(bucket[0])
            source_used.add(bucket[0]['record_id'])
        residual = sorted((row for row in source_rows if row['record_id'] not in source_used), key=lambda row: (strata(row), rank(row)))
        source_selected.extend(residual[:per_source - len(source_selected)])
        if len(source_selected) != per_source:
            raise ValueError(f'not enough rows for source {source}')
        selected.extend(source_selected)
        used.update(row['record_id'] for row in source_selected)
    if len(selected) != limit:
        raise ValueError('unable to fill queue')
    selected.sort(key=lambda row: (row['source'], row['path'], row['block_index'], row['record_id']))
    return selected


def build(input_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = input_dir / 'manifest.json'
    candidate_path = input_dir / 'candidate.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('status') != 'candidate_not_approved_for_training' or manifest.get('training_approval') is not False:
        raise ValueError('input is not an unapproved candidate')
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if manifest.get('candidate_sha256') != candidate_sha:
        raise ValueError('candidate hash mismatch')
    rows = [json.loads(line) for line in candidate_path.read_text(encoding='utf8').splitlines()]
    queue = select_queue(rows)
    output_dir.mkdir(parents=False)
    queue_path = output_dir / 'queue.jsonl'
    with queue_path.open('x', encoding='utf8', newline='\n') as stream:
        for index, row in enumerate(queue):
            stream.write(json.dumps({**row, 'queue_index': index, 'review_status': 'pending_independent_review'}, ensure_ascii=False, separators=(',', ':')) + '\n')
    result = {
        'status': 'review_queue_not_training_approval',
        'training_approval': False,
        'input_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'input_candidate_sha256': candidate_sha,
        'records': len(queue),
        'queue_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        'strata': dict(sorted(Counter('/'.join(strata(row)) for row in queue).items())),
        'selection_rule': 'one deterministic record per chapter, then stable rank fill; retained length and formula strata for review',
        'reviewer': 'pending independent review',
        'training_started': False,
        'limitations': ['Technical textbook only; not general Chinese or dialogue data.', 'Formula extraction is heuristic.', 'Queue creation is not training approval.'],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input, args.output)
    print(json.dumps({'output': str(args.output), 'records': result['records'], 'training_approval': result['training_approval']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
