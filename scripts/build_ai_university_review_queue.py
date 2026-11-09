"""Build a balanced review queue for the AI University candidate."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'runs/p185-ai-university-candidate-20261109'


def rank(row):
    return hashlib.sha256(('aiu-p186:' + row['record_id']).encode('utf8')).hexdigest()


def category(row):
    parts = row['source_path'].split('/')
    return parts[2] if len(parts) > 2 else parts[0]


def select(rows, limit=64):
    groups = {}
    for row in rows:
        groups.setdefault(category(row), []).append(row)
    categories = sorted(groups)
    per = limit // len(categories)
    if per < 1 or len(categories) > limit:
        raise ValueError('invalid category/queue size')
    chosen = []
    for name in categories:
        bucket = sorted(groups[name], key=rank)
        chosen.extend(bucket[:per])
    residual = sorted((row for row in rows if row not in chosen), key=rank)
    chosen.extend(residual[:limit - len(chosen)])
    if len(chosen) != limit:
        raise ValueError('unable to fill queue')
    return sorted(chosen, key=lambda row: (category(row), row['record_id']))


def build(input_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = input_dir / 'manifest.json'
    candidate_path = input_dir / 'candidate.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('status') != 'candidate_not_approved_for_training' or manifest.get('training_approval') is not False:
        raise ValueError('input is not an unapproved candidate')
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != manifest.get('candidate_sha256'):
        raise ValueError('candidate hash mismatch')
    rows = [json.loads(line) for line in candidate_path.read_text(encoding='utf8').splitlines()]
    queue = select(rows)
    output_dir.mkdir(parents=False)
    queue_path = output_dir / 'queue.jsonl'
    with queue_path.open('x', encoding='utf8', newline='\n') as stream:
        for index, row in enumerate(queue):
            stream.write(json.dumps({**row, 'queue_index': index, 'review_status': 'pending_independent_review'}, ensure_ascii=False, separators=(',', ':')) + '\n')
    result = {
        'status': 'review_queue_not_training_approval',
        'training_approval': False,
        'input_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'input_candidate_sha256': manifest['candidate_sha256'],
        'records': len(queue),
        'queue_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest(),
        'category_counts': dict(sorted(Counter(category(row) for row in queue).items())),
        'selection_rule': 'four deterministic records per category, then stable hash fill',
        'reviewer': 'pending independent review',
        'limitations': ['All lessons declare AI-generated and humanReviewed=false.', 'Queue is educational and structured, not general Chinese.', 'Queue creation is not training approval.'],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input, args.output)
    print(json.dumps({'output': str(args.output), 'records': result['records'], 'category_counts': result['category_counts'], 'training_approval': result['training_approval']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
