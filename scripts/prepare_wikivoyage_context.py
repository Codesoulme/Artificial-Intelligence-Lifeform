"""Build consecutive, page-grouped context candidates from a fixed dump."""
import argparse
import bz2
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from tokenizers import Tokenizer

from scripts.prepare_wikivoyage_paragraphs import SOURCE_SHA1, VOCAB_SHA256, digest, extract_paragraph

CANDIDATE_SHA = 'b22fa9febd9188de0c2d48c16503d5d24cf3cd8a656a7168feb81bff6f21b707'
OLD_SAMPLE_SHA = '7f60ba662e68df8171461c5a6c69c1de4efc857fb9f8b40e486360a546f4397e'
RISK = re.compile(r'-\{|\}-|^\s*(?:={2,}|[#*;:]|\{\||\|\})|'
                  r'(?:票价|票價|收费|收費|营业时间|營業時間|联系电话|聯絡電話)', re.I | re.M)
DEPENDENT = re.compile(r'^(?:其中|此外|因此|然而|但是|同时|同時|这座|這座|该市|該市|此处|此處)')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source, tokenizer_path, previous, output):
    source, tokenizer_path, previous, output = map(Path, (source, tokenizer_path, previous, output))
    if output.exists():
        raise FileExistsError(output)
    raw_candidate = (previous/'candidate.jsonl').read_bytes()
    raw_sample = (previous/'review-sample.json').read_bytes()
    if (digest(source, 'sha1') != SOURCE_SHA1 or digest(tokenizer_path, 'sha256') != VOCAB_SHA256
            or sha(raw_candidate) != CANDIDATE_SHA or sha(raw_sample) != OLD_SAMPLE_SHA):
        raise ValueError('Input digest mismatch')
    previous_rows = [json.loads(line) for line in raw_candidate.splitlines() if line.strip()]
    excluded = {row['page_id'] for row in json.loads(raw_sample)}
    per_page = defaultdict(dict)
    for row in previous_rows:
        if sha(row['text'].encode('utf8')) != row['text_sha256']:
            raise ValueError('Previous text hash mismatch')
        if row['page_id'] not in excluded:
            ordinal = row['paragraph_ordinal']
            if ordinal in per_page[row['page_id']]:
                raise ValueError('Duplicate page ordinal')
            per_page[row['page_id']][ordinal] = row
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    counts, result, seen_pages = Counter(), [], set()
    with bz2.open(source, 'rb') as stream:
        for _, page in ET.iterparse(stream, events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            ns = page.tag.split('}')[0]+'}'
            page_id = page.findtext(ns+'id')
            if page_id not in per_page:
                page.clear(); continue
            if page_id in seen_pages:
                raise ValueError('Duplicate source page')
            seen_pages.add(page_id)
            revision = page.find(ns+'revision')
            if revision is None:
                raise ValueError('Missing source revision')
            blocks = re.split(r'\n\s*\n', revision.findtext(ns+'text') or '')
            accepted = {}
            for ordinal, row in per_page[page_id].items():
                if (ordinal >= len(blocks) or revision.findtext(ns+'id') != row['revision_id']
                        or extract_paragraph(blocks[ordinal]) != row['text']):
                    raise ValueError('Source paragraph or revision mismatch')
                if RISK.search(blocks[ordinal]):
                    counts['format_or_time_risk'] += 1; continue
                accepted[ordinal] = row
            runs, current = [], []
            for ordinal in sorted(accepted):
                if current and ordinal != current[-1]['paragraph_ordinal'] + 1:
                    runs.append(current); current = []
                current.append(accepted[ordinal])
            if current:
                runs.append(current)
            for run in runs:
                if len(run) < 2 or DEPENDENT.match(run[0]['text']):
                    counts['short_or_dependent_runs'] += 1; continue
                text = '\n\n'.join(row['text'] for row in run)
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
                    counts['roundtrip_rejected'] += 1; continue
                result.append({'page_id': page_id, 'revision_id': run[0]['revision_id'],
                               'title': run[0]['title'], 'url': run[0]['url'], 'group': run[0]['group'],
                               'paragraph_ordinals': [r['paragraph_ordinal'] for r in run],
                               'paragraph_sha256': [r['text_sha256'] for r in run],
                               'text': text, 'text_sha256': sha(text.encode('utf8')),
                               'characters': len(text), 'content_tokens': len(ids)})
            page.clear()
    if seen_pages != set(per_page) or not result:
        raise ValueError('Missing source pages or no candidate')
    if digest(source, 'sha1') != SOURCE_SHA1 or digest(tokenizer_path, 'sha256') != VOCAB_SHA256:
        raise ValueError('Source changed during extraction')
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in result).encode('utf8')
    # One span per page avoids a long page dominating the review sample.
    unique_pages = {}
    for row in result:
        key = hashlib.sha256(('p223-context-sample-v1:'+row['text_sha256']).encode()).digest()
        if row['page_id'] not in unique_pages or key < unique_pages[row['page_id']][0]:
            unique_pages[row['page_id']] = (key, row)
    sample = [x[1] for x in heapq.nsmallest(min(40, len(unique_pages)), unique_pages.values(), key=lambda x:x[0])]
    summary = {'status': 'context_candidate_not_training_approved', 'source_sha1': SOURCE_SHA1,
               'parent_candidate_sha256': CANDIDATE_SHA, 'excluded_old_sample_sha256': OLD_SAMPLE_SHA,
               'tokenizer_sha256': VOCAB_SHA256, 'candidate_sha256': sha(raw),
               'counts': {**counts, 'excluded_previously_reviewed_pages': len(excluded),
                          'spans': len(result), 'pages': len(unique_pages),
                          'paragraphs': sum(len(r['paragraph_ordinals']) for r in result),
                          'characters': sum(r['characters'] for r in result),
                          'content_tokens': sum(r['content_tokens'] for r in result)},
               'sample_rule': 'one lowest-hash span per page, then lowest 40 page representatives',
               'test_split_read': False,
               'limitations': ['Rules and exclusions were informed by the earlier sample; new sample is for development only.',
                               'Consecutive paragraph ordinals do not prove self-contained text or correct extraction.',
                               'Costs, routes, factual and travel guidance may be outdated or unreliable.',
                               'No page-level rights, cross-source or semantic-overlap certification.',
                               'Candidate counts are not approved or high-quality training volume.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(raw)
    (output/'review-sample.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (output/'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tokenizer', 'previous', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.tokenizer, args.previous, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
