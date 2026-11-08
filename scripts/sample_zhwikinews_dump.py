"""Sample a fixed official news dump for local review, not model training."""
import bz2
import collections
import hashlib
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sample_zhwikibooks_dump import file_digest, plain_text

SOURCE = ROOT / 'runs/research/zhwikinews/zhwikinews-20261101-pages-articles.xml.bz2'
OUT = ROOT / 'runs/research/zhwikinews/train-audit-v1.json'
EXPECTED_SHA1 = 'f37cbb301e6bb965f8dd70eb3afdd628c370deab'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if file_digest(SOURCE, 'sha1') != EXPECTED_SHA1:
        raise ValueError('Dump differs from official checksum')
    counts, sample, rng = collections.Counter(), [], random.Random(851108)
    ns = None
    with bz2.open(SOURCE, 'rb') as f:
        for _, page in ET.iterparse(f, events=('end',)):
            if not page.tag.endswith('}page'):
                continue
            if ns is None:
                ns = page.tag.split('}')[0] + '}'
            counts['all_pages'] += 1
            if page.findtext(ns+'ns') != '0' or page.find(ns+'redirect') is not None:
                page.clear()
                continue
            title = page.findtext(ns+'title') or ''
            revision = page.find(ns+'revision')
            if not title or revision is None:
                counts['missing_title_or_revision'] += 1
                page.clear()
                continue
            title_hash = hashlib.sha256(title.encode('utf8')).hexdigest()
            if int(title_hash[:8], 16) % 10 == 0:
                counts['heldout_title_pages'] += 1
                page.clear()
                continue
            raw = revision.findtext(ns+'text') or ''
            value = plain_text(raw)
            if value is None or not 300 <= len(value) <= 8000:
                counts['format_or_length_reject'] += 1
                page.clear()
                continue
            chinese = sum('\u4e00' <= c <= '\u9fff' for c in value)
            if chinese / len(value) < .45:
                counts['language_reject'] += 1
                page.clear()
                continue
            counts['candidate_train_pages'] += 1
            counts['candidate_train_chars'] += len(value)
            item = dict(title=title, title_sha256=title_hash,
                        revision_id=revision.findtext(ns+'id'),
                        url='https://zh.wikinews.org/wiki/'+quote(title.replace(' ', '_'), safe='/'),
                        raw_sha256=hashlib.sha256(raw.encode('utf8')).hexdigest(),
                        text_sha256=hashlib.sha256(value.encode('utf8')).hexdigest(),
                        characters=len(value), text=value)
            if len(sample) < 40:
                sample.append(item)
            else:
                index = rng.randrange(counts['candidate_train_pages'])
                if index < 40:
                    sample[index] = item
            page.clear()
    if len(sample) != 40:
        raise ValueError('Insufficient candidates for audit')
    report = dict(status='sample_not_approved_for_training', source_sha1=EXPECTED_SHA1,
                  source_sha256=file_digest(SOURCE, 'sha256'),
                  generator_sha256=file_digest(Path(__file__), 'sha256'),
                  extraction_helper_sha256=file_digest(ROOT/'scripts/sample_zhwikibooks_dump.py', 'sha256'),
                  seed=851108, counts=counts, samples=sample,
                  limitations=['Training bucket by title hash does not prevent related articles, translations or near duplicates crossing splits.',
                               'Wiki markup, templates and references are removed; output can lose facts or context.',
                               'Articles may be dated, sensitive, quoted, incomplete, or incorrectly licensed.',
                               'Sample is not a training approval and cannot establish full-corpus quality.'])
    with OUT.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
