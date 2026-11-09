"""Select the explicitly reviewed training-side sample without expanding approval."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def select(candidate_dir, review_path, output):
    root, review_path, output = Path(candidate_dir), Path(review_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
    candidate = (root/'candidate.jsonl').read_bytes()
    sample_raw = (root/'review-sample.json').read_bytes()
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    sample = json.loads(sample_raw)
    if (sha(candidate) != manifest['candidate_sha256'] or sha(candidate) != review['candidate_sha256']
            or sha(sample_raw) != review['sample_sha256']
            or review.get('status') != 'model_review_not_training_approval'
            or len(sample) != review['sample_size']):
        raise ValueError('Candidate or review identity mismatch')
    fields = ('pass_indices', 'uncertain_indices', 'reject_indices')
    indices = [i for field in fields for i in review[field]]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(sample))):
        raise ValueError('Review must cover the sample exactly once')
    rows = [json.loads(line) for line in candidate.splitlines() if line.strip()]
    by_hash = {r['text_sha256']: r for r in rows}
    if len(by_hash) != len(rows):
        raise ValueError('Duplicate candidate hashes')
    for row in sample:
        if sha(row['text'].encode('utf8')) != row['text_sha256'] or by_hash.get(row['text_sha256']) != row:
            raise ValueError('Sample record mismatch')
    selected = [sample[i] for i in review['pass_indices']]
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in selected).encode('utf8')
    result = {'status': 'model_reviewed_sample_not_training_approved',
              'source_candidate_sha256': sha(candidate), 'sample_sha256': sha(sample_raw),
              'review_sha256': sha(review_raw), 'output_sha256': sha(raw),
              'paragraphs': len(selected), 'pages': len({r['page_id'] for r in selected}),
              'characters': sum(r['characters'] for r in selected),
              'content_tokens': sum(r['content_tokens'] for r in selected),
              'test_split_read': False,
              'limitations': ['Review was by models, not human or independent factual verification.',
                              'Attribution, page history, third-party material and rights remain unresolved.',
                              'This is a small sample, not an estimate of usable volume.',
                              'Historical and final-test semantic overlap is not ruled out.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(raw)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'review', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(select(args.candidate, args.review, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
