"""Bind captured story pages to pinned source text; never approve training."""
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/research/storybooks-canada'
ALLOWED = {f'https://creativecommons.org/licenses/by/{v}/' for v in ('3.0', '4.0')}


class Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.story_depth = None
        self.h3_depth = None
        self.parts = []
        self.result = []

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            self.depth += 1
            classes = dict(attrs).get('class', '').split()
            if any(re.fullmatch(r'level\d+-txt', c) for c in classes) and 'def' in classes:
                if self.story_depth is not None:
                    raise ValueError('Nested story text')
                self.story_depth = self.depth
        if tag == 'h3' and self.story_depth is not None:
            if self.h3_depth is not None:
                raise ValueError('Nested story heading')
            self.h3_depth = self.depth
            self.parts = []

    def handle_data(self, data):
        if self.h3_depth is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'h3' and self.h3_depth is not None:
            self.result.append(''.join(self.parts))
            self.h3_depth = None
        if tag == 'div':
            if self.story_depth == self.depth:
                self.story_depth = None
            self.depth -= 1


def normalise(text):
    return re.sub(r'\s+', '', text)


def verify(row):
    result = {'story': row['story'], 'source_sha256': row['source_sha256']}
    if row['source_label'] != 'CC-BY':
        result['status'] = 'excluded_source_label'
        return result
    source = (BASE / 'raw-v1' / (row['story'] + '.md')).read_bytes()
    page = (BASE / 'license-audit-v1' / (row['story'] + '.html')).read_bytes()
    if hashlib.sha256(source).hexdigest() != row['source_sha256'] or hashlib.sha256(page).hexdigest() != row['page_sha256']:
        raise ValueError('Source or page hash mismatch: ' + row['story'])
    text = source.decode('utf-8')
    segments = re.split(r'^##\s*$', text, flags=re.M)[1:]
    if not segments:
        raise ValueError('Missing source sections: ' + row['story'])
    metadata = segments.pop()
    attribution = {}
    for key in ('Text', 'Illustration', 'Translation', 'License'):
        values = re.findall(r'^\*\s*' + key + r':\s*(.+?)\s*$', metadata, re.M)
        if len(values) != 1:
            raise ValueError('Missing or ambiguous attribution: ' + row['story'] + ' ' + key)
        attribution[key] = values[0]
    parser = Paragraphs()
    parser.feed(page.decode('utf-8'))
    source_paragraphs = [normalise(p) for p in segments]
    page_paragraphs = [normalise(p) for p in parser.result]
    matching = bool(source_paragraphs) and source_paragraphs == page_paragraphs
    license_ok = len(row['license_urls']) == 1 and row['license_urls'][0] in ALLOWED
    result.update(page_url=row['page_url'], page_sha256=row['page_sha256'],
                  source_path=row['source_path'], license_urls=row['license_urls'],
                  attribution=attribution, paragraphs=len(source_paragraphs),
                  text_matches_page=matching, allowed_page_license=license_ok,
                  status='matched_pending_quality_and_rights_review' if matching and license_ok else 'review_required')
    return result


def main():
    manifest = json.loads((BASE / 'license-audit-v1/manifest.json').read_text(encoding='utf-8'))
    results = [verify(row) for row in manifest['stories']]
    out = BASE / 'page-text-verification-v2.json'
    report = {'status': 'not_approved_for_training', 'source_revision': manifest['revision'],
              'page_capture_time_utc': manifest['reviewed_at_utc'], 'stories': results,
              'limitations': ['Whitespace-normalised paragraph comparison; no translation accuracy or upstream rights validation.',
                              'Captured pages can change; per-work attribution and rights still require review.']}
    with out.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({s: sum(r['status'] == s for r in results) for s in sorted({r['status'] for r in results})}))


if __name__ == '__main__':
    main()
