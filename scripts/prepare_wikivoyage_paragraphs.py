"""Extract traceable training-side paragraph candidates from a fixed dump."""
import argparse
import bz2
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote

import mwparserfromhell
from tokenizers import Tokenizer

SOURCE_SHA1 = 'f8e4b3c9f88b74aeffb5427d97b9d2712f0761a4'
VOCAB_SHA256 = 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'
SALT = 'p222-wikivoyage-page-v1:'
RISK = re.compile(r'\{\{|\}\}|<|>|\{\||\|\}|https?://|www\.|\b\S+@\S+\b|'
                  r'(?:电话|電話|手机|手機|地址|营业时间|營業時間|邮编|郵編)|'
                  r'\d[\d ()+\-]{7,}\d|\[\[(?:file|image|category|文件|檔案|分类|分類):', re.I)


def digest(path, algorithm):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def extract_paragraph(raw):
    raw = raw.strip()
    if not raw or raw[0] in '#*;:=!{|[' or RISK.search(raw):
        return None
    code = mwparserfromhell.parse(raw)
    if code.filter_templates(recursive=True) or code.filter_tags(recursive=True) or code.filter_external_links():
        return None
    text = re.sub(r'\s+', ' ', code.strip_code(normalize=True, collapse=True)).strip()
    if not 80 <= len(text) <= 1500 or sum('\u4e00' <= c <= '\u9fff' for c in text) / len(text) < .5:
        return None
    if '\ufffd' in text or not re.search(r'[。！？][”’」』）)]?$', text):
        return None
    return text


def prepare(source, tokenizer_path, output):
    source, tokenizer_path, output = map(Path, (source, tokenizer_path, output))
    if output.exists():
        raise FileExistsError(output)
    if digest(source, 'sha1') != SOURCE_SHA1 or digest(tokenizer_path, 'sha256') != VOCAB_SHA256:
        raise ValueError('Source or tokenizer digest mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    counts = Counter()
    rows, hashes, groups = [], set(), set()
    with bz2.open(source, 'rb') as stream:
        for _, page in ET.iterparse(stream, events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            ns = page.tag.split('}')[0] + '}'
            if page.findtext(ns+'ns') != '0' or page.find(ns+'redirect') is not None:
                page.clear(); continue
            counts['main_pages'] += 1
            page_id, title = page.findtext(ns+'id'), page.findtext(ns+'title')
            revision = page.find(ns+'revision')
            if not page_id or not title or revision is None or not revision.findtext(ns+'id'):
                counts['missing_metadata'] += 1; page.clear(); continue
            group = hashlib.sha256((SALT+page_id).encode('utf8')).hexdigest()
            if int(group[:8], 16) % 10 == 0:
                counts['reserved_pages'] += 1; page.clear(); continue
            wikitext = revision.findtext(ns+'text') or ''
            for ordinal, paragraph in enumerate(re.split(r'\n\s*\n', wikitext)):
                counts['paragraphs_seen'] += 1
                text = extract_paragraph(paragraph)
                if text is None:
                    continue
                text_hash = hashlib.sha256(text.encode('utf8')).hexdigest()
                if text_hash in hashes:
                    counts['exact_duplicates'] += 1; continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids or tokenizer.decode(ids, skip_special_tokens=False) != text:
                    counts['tokenizer_rejected'] += 1; continue
                hashes.add(text_hash); groups.add(group)
                rows.append({'page_id': page_id, 'revision_id': revision.findtext(ns+'id'),
                             'title': title, 'url': 'https://zh.wikivoyage.org/wiki/'+quote(title.replace(' ', '_'), safe='/'),
                             'group': group, 'paragraph_ordinal': ordinal, 'text': text,
                             'text_sha256': text_hash, 'characters': len(text), 'content_tokens': len(ids)})
            page.clear()
    if not rows:
        raise ValueError('No training-side paragraphs')
    if digest(source, 'sha1') != SOURCE_SHA1 or digest(tokenizer_path, 'sha256') != VOCAB_SHA256:
        raise ValueError('Input changed during extraction')
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in rows).encode('utf8')
    sample = heapq.nsmallest(min(40, len(rows)), rows,
                            key=lambda row: hashlib.sha256(('p222-review-v1:'+row['text_sha256']).encode()).digest())
    result = {'status': 'mechanical_candidate_not_training_approved', 'source_sha1': SOURCE_SHA1,
              'source_sha256': digest(source, 'sha256'), 'tokenizer_sha256': VOCAB_SHA256,
              'dump_url': 'https://dumps.wikimedia.org/zhwikivoyage/20261101/zhwikivoyage-20261101-pages-articles.xml.bz2',
              'license_declared': 'CC BY-SA 4.0; individual page and third-party rights not reviewed',
              'reserved_rule': 'page id SHA256 with fixed salt, first 8 hex digits modulo 10 equals zero',
              'counts': {**counts, 'candidate_paragraphs': len(rows), 'candidate_pages': len(groups),
                         'characters': sum(r['characters'] for r in rows),
                         'content_tokens': sum(r['content_tokens'] for r in rows)},
              'sample_rule': 'lowest SHA256(p222-review-v1: + paragraph SHA256), at most 40 paragraphs',
              'candidate_sha256': hashlib.sha256(raw).hexdigest(), 'test_split_read': False,
              'limitations': ['Paragraph removal may break context; retained links are rendered as display text.',
                              'Phone and address filters are incomplete; facts and travel advice may be stale.',
                              'Pages, translations, source overlap and license obligations remain unreviewed.',
                              'Page-id buckets do not exclude aliases, related pages or semantic overlap.',
                              'Numbers are mechanical candidates, not verified quality or training approval.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(raw)
    (output/'review-sample.json').write_text(json.dumps(sample, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tokenizer', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.tokenizer, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
