"""Build a deterministic 64-record quality review queue; never trains."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'runs/p172-chinesewebtext-range-candidate-20261109'


def sha256_path(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(record_id):
    return hashlib.sha256(('cw2-p173-review:' + record_id).encode('utf8')).hexdigest()


def select_queue(rows, limit=64, per_domain=6):
    domains = sorted({row['domain']['single_label'] for row in rows})
    selected = []
    used = set()
    for domain in domains:
        bucket = sorted((row for row in rows if row['domain']['single_label'] == domain), key=lambda row: rank(row['record_id']))
        for row in bucket[:min(per_domain, len(bucket))]:
            selected.append(row)
            used.add(row['record_id'])
    if len(selected) > limit:
        raise ValueError('per-domain quota exceeds queue size')
    if len(selected) < limit:
        residual = sorted((row for row in rows if row['record_id'] not in used), key=lambda row: rank(row['record_id']))
        selected.extend(residual[:limit - len(selected)])
    if len(selected) != limit or len({row['record_id'] for row in selected}) != limit:
        raise ValueError('unable to construct distinct queue')
    selected.sort(key=lambda row: (row['domain']['single_label'], row['record_id']))
    return selected


def build(input_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest_path = input_dir / 'manifest.json'
    candidate_path = input_dir / 'candidate.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    if manifest.get('status') != 'candidate_not_approved_for_training' or manifest.get('training_approval') is not False:
        raise ValueError('input is not an unapproved candidate')
    if manifest.get('candidate_sha256') != sha256_path(candidate_path):
        raise ValueError('candidate hash mismatch')
    rows = [json.loads(line) for line in candidate_path.read_text(encoding='utf8').splitlines()]
    if len(rows) != manifest.get('records') or len(rows) != 500:
        raise ValueError('unexpected candidate size')
    queue = select_queue(rows)
    output_dir.mkdir(parents=False)
    queue_path = output_dir / 'queue.jsonl'
    with queue_path.open('x', encoding='utf8', newline='\n') as stream:
        for index, row in enumerate(queue):
            stream.write(json.dumps({
                'queue_index': index,
                'record_id': row['record_id'],
                'source_line_number': row['source_line_number'],
                'domain': row['domain'],
                'quality_score': row['quality_score'],
                'toxicity': row['toxicity'],
                'text': row['text'],
                'review_status': 'pending_independent_review',
            }, ensure_ascii=False, separators=(',', ':')) + '\n')
    result = {
        'status': 'review_queue_not_training_approval',
        'training_approval': False,
        'input_manifest_sha256': sha256_path(manifest_path),
        'input_candidate_sha256': manifest['candidate_sha256'],
        'records': len(queue),
        'queue_sha256': sha256_path(queue_path),
        'selection_rule': 'six deterministic SHA256-ranked records per eligible domain, then fill remaining slots by the same rank; no text quality decision made',
        'domain_counts': dict(sorted(Counter(row['domain']['single_label'] for row in queue).items())),
        'reviewer': 'pending independent review',
        'limitations': [
            'Queue is drawn from one 16 MiB range of one train-side shard.',
            'Publisher scores are retained as metadata and are not review labels.',
            'Queue creation does not approve the 500-record candidate or authorize training.',
        ],
    }
    with (output_dir / 'manifest.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.input, args.output)
    print(json.dumps({
        'output': str(args.output),
        'records': result['records'],
        'domain_counts': result['domain_counts'],
        'training_approval': result['training_approval'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
