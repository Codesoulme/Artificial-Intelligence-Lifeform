"""Prepare unseen-work extraction review pages from the fixed P225 candidate."""
import argparse
import bz2
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.sample_zhwikibooks_dump import EXPECTED_SHA1, file_digest
from scripts.wikibooks_structured import extract

CANDIDATE_SHA = '1cd27640cdd3c29e37df938d2ef08a8e7187eceee7a8bd8dd41137dc75e2ba23'
OLD_SAMPLE_SHA = 'd56e0b6bb4eae04dd0d2b2cee75c975900b95e7271778d4a310c6a1256277280'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source, candidate, old_sample, output, count=30):
    source, candidate, old_sample, output = map(Path, (source, candidate, old_sample, output))
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or count < 1:
        raise ValueError('Invalid sample size')
    candidate_raw, old_raw = candidate.read_bytes(), old_sample.read_bytes()
    if (file_digest(source, 'sha1') != EXPECTED_SHA1 or sha(candidate_raw) != CANDIDATE_SHA
            or sha(old_raw) != OLD_SAMPLE_SHA):
        raise ValueError('Input digest mismatch')
    excluded = {r['work_sha256'] for r in json.loads(old_raw)}
    representatives = {}
    for line in candidate_raw.splitlines():
        row = json.loads(line)
        group = row['work_sha256']
        if group in excluded:
            continue
        rank = sha(('p226-page:'+row['page_id']).encode())
        if group not in representatives or rank < representatives[group][0]:
            representatives[group] = (rank, row)
    ordered = sorted(representatives.items(), key=lambda x: sha(('p226-work:'+x[0]).encode()))[:count]
    selected = [item[1][1] for item in ordered]
    if len(selected) != count or len({r['page_id'] for r in selected}) != count:
        raise ValueError('Insufficient or duplicate review pages')
    pages = {r['page_id']: r for r in selected}
    found = {}
    with bz2.open(source, 'rb') as stream:
        for _, page in ET.iterparse(stream, events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            ns = page.tag.split('}')[0]+'}'
            page_id = page.findtext(ns+'id')
            if page_id not in pages:
                page.clear(); continue
            if page_id in found:
                raise ValueError('Duplicate page')
            revision = page.find(ns+'revision')
            if revision is None:
                raise ValueError('Missing revision')
            text = revision.findtext(ns+'text') or ''
            if revision.findtext(ns+'id') != pages[page_id]['revision_id'] or sha(text.encode('utf8')) != pages[page_id]['raw_sha256']:
                raise ValueError('Page revision or content mismatch')
            found[page_id] = text
            page.clear()
    if set(found) != set(pages) or file_digest(source, 'sha1') != EXPECTED_SHA1:
        raise ValueError('Missing pages or changed source')
    sample = []
    for index, row in enumerate(selected):
        raw = found[row['page_id']]
        sample.append({'index': index, 'page_id': row['page_id'], 'title': row['title'],
                       'revision_id': row['revision_id'], 'work_sha256': row['work_sha256'],
                       'raw_sha256': row['raw_sha256'], 'raw_wikitext': raw,
                       'extraction': extract(raw)})
    sample_raw = (json.dumps(sample, ensure_ascii=False, indent=2)+'\n').encode('utf8')
    result = {'status': 'unseen_work_extraction_validation_not_training_approval',
              'source_sha1': EXPECTED_SHA1, 'candidate_sha256': CANDIDATE_SHA,
              'previous_sample_sha256': OLD_SAMPLE_SHA, 'sample_sha256': sha(sample_raw),
              'pages': len(sample), 'excluded_previous_work_prefixes': len(excluded),
              'limitations': ['Title prefixes may not capture related works, translations or aliases.',
                              'The XML parser loads all pages prior to selecting these work groups.',
                              'Validation targets extraction and does not certify factual accuracy or rights.']}
    output.mkdir(parents=False)
    (output/'review-sample.json').write_bytes(sample_raw)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'candidate', 'old-sample', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.candidate, args.old_sample, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
