"""Prepare a traceable training-side Wikibooks review pool, not training data."""
import argparse
import bz2
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

from tokenizers import Tokenizer
from scripts.sample_zhwikibooks_dump import EXPECTED_SHA1, file_digest, plain_text

VOCAB_SHA256 = 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'
AUDIT_SHA256 = 'fb869de5799167d42de692c7b4ca6254f2b8b3c9bb80ecd690557918e01ddd71'
RISK = re.compile(r'\{\{|\[\[(?:File|Image|文件|檔案|图像|圖像):|<\s*(?:ref|gallery)\b|'
                  r'-\{|https?://|\{\||^\s*[*#;:]', re.I | re.M)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(source, tokenizer_path, old_audit, output, sample_size=40):
    source, tokenizer_path, old_audit, output = map(Path, (source, tokenizer_path, old_audit, output))
    if output.exists():
        raise FileExistsError(output)
    if type(sample_size) is not int or sample_size < 1:
        raise ValueError('Invalid sample size')
    if file_digest(source, 'sha1') != EXPECTED_SHA1 or file_digest(tokenizer_path, 'sha256') != VOCAB_SHA256:
        raise ValueError('Source or tokenizer digest mismatch')
    audit_raw = old_audit.read_bytes()
    if sha(audit_raw) != AUDIT_SHA256:
        raise ValueError('Previous audit digest mismatch')
    audit = json.loads(audit_raw)
    if audit.get('source_sha1') != EXPECTED_SHA1 or len(audit.get('samples', [])) != audit.get('sample_size'):
        raise ValueError('Previous audit does not match source')
    excluded = {row['work_sha256'] for row in audit['samples']}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows, counts, seen, raw_pages = [], Counter(), set(), {}
    with bz2.open(source, 'rb') as stream:
        for _, page in ET.iterparse(stream, events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            ns = page.tag.split('}')[0]+'}'
            if page.findtext(ns+'ns') != '0' or page.find(ns+'redirect') is not None:
                page.clear(); continue
            title = page.findtext(ns+'title') or ''
            work_hash = sha(title.split('/', 1)[0].encode('utf8'))
            # Do not extract P84's work-level holdout or previously inspected works.
            if not title or int(work_hash[:8], 16) % 10 == 0 or work_hash in excluded:
                page.clear(); continue
            revision = page.find(ns+'revision')
            page_id = page.findtext(ns+'id')
            if revision is None or not page_id or not revision.findtext(ns+'id') or page_id in seen:
                raise ValueError('Missing or duplicate page identity')
            seen.add(page_id)
            raw = revision.findtext(ns+'text') or ''
            text = plain_text(raw)
            if text is None or not 300 <= len(text) <= 8000:
                counts['format_or_length_rejected'] += 1; page.clear(); continue
            if sum('\u4e00' <= c <= '\u9fff' for c in text) / len(text) < .45:
                counts['language_rejected'] += 1; page.clear(); continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
                counts['roundtrip_rejected'] += 1; page.clear(); continue
            rows.append({'page_id': page_id, 'revision_id': revision.findtext(ns+'id'),
                         'revision_timestamp': revision.findtext(ns+'timestamp'),
                         'title': title, 'work_sha256': work_hash,
                         'url': 'https://zh.wikibooks.org/wiki/'+quote(title.replace(' ', '_'), safe='/'),
                         'revision_url': 'https://zh.wikibooks.org/w/index.php?oldid='+revision.findtext(ns+'id'),
                         'raw_sha256': sha(raw.encode('utf8')), 'text_sha256': sha(text.encode('utf8')),
                         'text': text, 'characters': len(text), 'content_tokens': len(ids),
                         'raw_markup_risk': bool(RISK.search(raw))})
            raw_pages[page_id] = raw
            page.clear()
    if not rows:
        raise ValueError('No candidate pages')
    # Choose a representative uniformly by stable hash within each work, then sample works.
    groups = {}
    for row in rows:
        rank = sha(('p225-page:'+row['page_id']).encode())
        current = groups.get(row['work_sha256'])
        if current is None or rank < current[0]:
            groups[row['work_sha256']] = (rank, row)
    ordered = sorted(groups.items(), key=lambda item: sha(('p225-work:'+item[0]).encode()))
    sample = [{**pair[1][1], 'raw_wikitext': raw_pages[pair[1][1]['page_id']]}
              for pair in ordered[:sample_size]]
    candidate_raw = ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':'))+'\n' for r in rows).encode('utf8')
    sample_raw = (json.dumps(sample, ensure_ascii=False, indent=2)+'\n').encode('utf8')
    if file_digest(source, 'sha1') != EXPECTED_SHA1 or file_digest(tokenizer_path, 'sha256') != VOCAB_SHA256:
        raise ValueError('Input changed during extraction')
    result = {'status': 'review_candidate_not_approved_for_training', 'source_sha1': EXPECTED_SHA1,
              'tokenizer_sha256': VOCAB_SHA256, 'previous_audit_sha256': sha(audit_raw),
              'candidate_sha256': sha(candidate_raw), 'sample_sha256': sha(sample_raw),
              'sampling': 'one lowest-hash page per work, then lowest-hash works',
              'counts': {**counts, 'excluded_previous_sample_works': len(excluded), 'pages': len(rows),
                         'works': len(groups), 'characters': sum(r['characters'] for r in rows),
                         'content_tokens': sum(r['content_tokens'] for r in rows),
                         'raw_markup_risk_pages': sum(r['raw_markup_risk'] for r in rows),
                         'sample_pages': len(sample)},
              'limitations': ['Extraction removes templates and references and normalizes whitespace; original context may be lost.',
                              'The XML parser loads each page before holdout filtering, although held-out text is not extracted or saved.',
                              'Raw markup flag is not a comprehensive risk or quality filter.',
                              'Work grouping by title prefix does not identify aliases, translations or shared sources.',
                              'Sample only represents the selected work/page scheme, not page-weighted quality.',
                              'Rights, facts, privacy and overlap with other training material are not verified.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(candidate_raw)
    (output/'review-sample.json').write_bytes(sample_raw)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tokenizer', 'old-audit', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.tokenizer, args.old_audit, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
