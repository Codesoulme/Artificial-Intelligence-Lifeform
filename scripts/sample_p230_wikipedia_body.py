"""Sample one candidate passage per page for bounded local review."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(candidate_dir, output, count=40):
    candidate_dir, output = Path(candidate_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or count < 1:
        raise ValueError('Invalid sample count')
    manifest_raw = (candidate_dir/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    raw = (candidate_dir/'candidates.jsonl').read_bytes()
    if (manifest.get('status') != 'pending_review_not_approved_for_training'
            or sha(raw) != manifest.get('files', {}).get('candidates.jsonl')):
        raise ValueError('Candidate manifest or digest mismatch')
    page_choices, seen, total = {}, set(), 0
    for line in raw.splitlines():
        row = json.loads(line)
        total += 1
        text = row['text']
        if sha(text.encode('utf8')) != row['text_sha256']:
            raise ValueError('Passage digest mismatch')
        identity = (row['page_id'], row['char_start'], row['char_end'])
        if (identity in seen or not (0 <= row['char_start'] < row['char_end'])
                or row['char_end'] - row['char_start'] != len(text)):
            raise ValueError('Duplicate or invalid passage position')
        seen.add(identity)
        rank = sha(('p230-sample-span:'+row['page_id']+':'+str(row['char_start'])).encode())
        if row['page_id'] not in page_choices or rank < page_choices[row['page_id']][0]:
            page_choices[row['page_id']] = (rank, row)
    if total != manifest['counts']['passages'] or len(page_choices) < count:
        raise ValueError('Candidate count or sample size mismatch')
    pages = sorted(page_choices.items(), key=lambda pair: sha(('p230-sample-page:'+pair[0]).encode()))
    sample = [pair[1][1] for pair in pages[:count]]
    sample_raw = (json.dumps(sample, ensure_ascii=False, indent=2)+'\n').encode('utf8')
    result = {'status': 'training_side_sample_pending_review', 'candidate_sha256': sha(raw),
              'candidate_manifest_sha256': sha(manifest_raw), 'sample_sha256': sha(sample_raw),
              'sampling': 'one lowest-hash passage per page, then lowest-hash pages',
              'candidate_pages': len(page_choices), 'sample_pages': len(sample),
              'limitations': ['Sample is page-weighted, not passage-weighted.',
                              'One sampled passage cannot establish quality of the rest of its page.',
                              'Additional page context, rights and source overlap require separate review.']}
    output.mkdir(parents=False)
    (output/'sample.json').write_bytes(sample_raw)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.candidate, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
