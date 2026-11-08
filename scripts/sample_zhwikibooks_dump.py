"""Inspect a fixed Chinese Wikibooks dump; writes only a local audit sample."""
import bz2
import collections
import hashlib
import json
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import mwparserfromhell

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/research/zhwikibooks/pages-official.xml.bz2'
OUT = ROOT/'runs/research/zhwikibooks/train-audit-v1.json'
EXPECTED_SHA1 = '905823ff0d8963bf676409008238e3ab46c3fa42'
RISK_FORMAT = re.compile(r'<\s*(?:math|chem|syntaxhighlight|source|gallery|table)\b|\{\|',re.I)


def file_digest(path, kind):
    with path.open('rb') as f:
        return hashlib.file_digest(f,kind).hexdigest()


def plain_text(value):
    if RISK_FORMAT.search(value):
        return None
    code = mwparserfromhell.parse(value)
    for tag in list(code.filter_tags(recursive=True)):
        if str(tag.tag).lower() in ('ref','references'):
            try:
                code.remove(tag,recursive=True)
            except ValueError:
                # A containing reference may already have been removed.
                pass
    for template in list(code.filter_templates(recursive=False)):
        code.remove(template,recursive=False)
    text = unicodedata.normalize('NFKC',code.strip_code(normalize=True,collapse=True))
    return re.sub(r'\s+',' ',text).strip()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_digest(SOURCE,'sha1') != EXPECTED_SHA1:
        raise ValueError('Official dump checksum mismatch')
    counts, sample, rng = collections.Counter(), [], random.Random(841109)
    ns = None
    with bz2.open(SOURCE,'rb') as f:
        for event, page in ET.iterparse(f,events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            if ns is None:
                ns = page.tag.split('}')[0]+'}'
            counts['all_pages'] += 1
            if page.findtext(ns+'ns') != '0' or page.find(ns+'redirect') is not None:
                page.clear();continue
            counts['main_nonredirect_pages'] += 1
            title = page.findtext(ns+'title') or ''
            revision = page.find(ns+'revision')
            if revision is None or not title:
                counts['missing_revision'] += 1;page.clear();continue
            work = title.split('/',1)[0]
            group = hashlib.sha256(work.encode('utf8')).hexdigest()
            if int(group[:8],16)%10 == 0:
                counts['heldout_work_pages'] += 1;page.clear();continue
            value = revision.findtext(ns+'text') or ''
            text = plain_text(value)
            if text is None or not 300 <= len(text) <= 8000:
                counts['format_or_length_reject'] += 1;page.clear();continue
            chinese = sum('\u4e00' <= c <= '\u9fff' for c in text)
            if chinese/len(text) < .45:
                counts['language_reject'] += 1;page.clear();continue
            counts['candidate_train_pages'] += 1
            counts['candidate_train_chars'] += len(text)
            item = dict(work_sha256=group,title=title,revision_id=revision.findtext(ns+'id'),
                        url='https://zh.wikibooks.org/wiki/'+quote(title.replace(' ','_'),safe='/'),
                        raw_sha256=hashlib.sha256(value.encode('utf8')).hexdigest(),
                        characters=len(text),text_prefix=text[:1200])
            if len(sample)<40:
                sample.append(item)
            else:
                i = rng.randrange(counts['candidate_train_pages'])
                if i<40:sample[i]=item
            page.clear()
    if len(sample)<40:
        raise ValueError('Not enough eligible training pages')
    result = dict(status='sample_not_approved_for_training', source_sha1=EXPECTED_SHA1,
                  source_sha256=file_digest(SOURCE,'sha256'),generator_sha256=file_digest(Path(__file__),'sha256'),
                  seed=841109,counts=counts,sample_size=len(sample),samples=sample,
                  limitations=['Text extraction removes templates and reference tags; meaning may be lost.',
                               'Sample only covers first 1200 characters, not full-page quality.',
                               'Book-level grouping by title prefix does not detect aliases or related books.',
                               'Page-level source rights and attribution still require review.',
                               'No training approval or final-test claim.'])
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False))


if __name__=='__main__':
    main()
