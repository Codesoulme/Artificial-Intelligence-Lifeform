"""Build a local review candidate; no training, no test-set access."""
import argparse
import hashlib
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'runs/research/storybooks-canada'
AUDIT = ROOT / 'runs/storybooks-dedup-diagnostic-20261108/results.json'
VOCAB = ROOT / 'runs/conversation-data-v5/tokenizer.json'
SALT = 'storybookscanada-zh-split-v1:'
ALLOWED = {f'https://creativecommons.org/licenses/by/{v}/' for v in ('3.0','4.0')}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def readjson(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def keyed(rows):
    result = {x['story']: x for x in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate story')
    return result

def split_for(story):
    digest = sha((SALT + story).encode('utf-8'))
    return ('dev' if int(digest, 16) % 10 == 0 else 'train', digest)

def extract(raw, count):
    sections = re.split(r'(?m)^##[ \t]*$', raw)
    if len(sections) != count + 2:
        raise ValueError('unexpected section count')
    title = sections[0].strip()
    if not title.startswith('# '):
        raise ValueError('missing title')
    paragraphs = [s.strip('\r\n') for s in sections[1:-1]]
    if not all(p.strip() for p in paragraphs):
        raise ValueError('empty paragraph')
    meta = sections[-1]
    attrs = {}
    for key in ('Text', 'Illustration', 'Translation', 'License', 'Language'):
        found = re.findall(r'^\*\s*' + key + r':\s*(.+?)\s*$', meta, re.M)
        if len(found) != 1:
            raise ValueError('missing/ambiguous attribution ' + key)
        attrs[key] = found[0]
    if attrs['License'] != '[CC-BY]' or attrs['Language'] != 'zh':
        raise ValueError('source label mismatch')
    return title[2:], paragraphs, attrs

class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.story_depth = None
        self.heading = False
        self.parts = []
        self.paragraphs = []
        self.licenses = set()
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'a' and 'license' in a.get('rel','').split():
            self.licenses.add(a.get('href'))
        if tag == 'div':
            self.depth += 1
            c = a.get('class','').split()
            if 'def' in c and any(re.fullmatch(r'level\d+-txt', x) for x in c):
                if self.story_depth is not None: raise ValueError('nested story')
                self.story_depth = self.depth
        if tag == 'h3' and self.story_depth is not None:
            if self.heading: raise ValueError('nested heading')
            self.heading = True
            self.parts = []
    def handle_data(self, data):
        if self.heading: self.parts.append(data)
    def handle_endtag(self, tag):
        if tag == 'h3' and self.heading:
            self.paragraphs.append(''.join(self.parts)); self.heading = False
        if tag == 'div':
            if self.story_depth == self.depth: self.story_depth = None
            self.depth -= 1

def compact(s):
    return re.sub(r'\s+', '', s)

def normalize(s):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', s).casefold())

def overlaps(texts):
    norm = [normalize(x) for x in texts]
    result = []
    for i, text in enumerate(norm):
        peers = [n for j,n in enumerate(norm) if j != i]
        item = {'full_containment_peers': sum(text in p for p in peers)}
        for n in (12, 20):
            windows = {text[k:k+n] for k in range(len(text)-n+1)}
            item[str(n)+'_peers'] = sum(any(w in p for w in windows) for p in peers)
        result.append(item)
    return result

def output_path(name, runs=ROOT / 'runs'):
    """Only create a new direct child of runs, never an existing run."""
    if not name or name in ('.', '..') or Path(name).name != name or '/' in name or '\\' in name or ':' in name:
        raise ValueError('output must be a single new directory name under runs')
    return runs / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', required=True, help='new directory name under runs/')
    args = parser.parse_args()
    out = output_path(args.output_name)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'output already exists: {out}')
    paths = {'page_verification': BASE/'page-text-verification-v2.json',
             'review_a': BASE/'quality-review-a.json', 'review_b': BASE/'quality-review-b.json',
             'review_independent': BASE/'quality-review-independent.json',
             'license_manifest': BASE/'license-audit-v1/manifest.json', 'historical_audit': AUDIT,
             'tokenizer': VOCAB}
    inputs = {k: {'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p.read_bytes())} for k,p in paths.items()}
    verification = keyed(readjson(paths['page_verification'])['stories'])
    first = keyed(readjson(paths['review_a'])['stories'] + readjson(paths['review_b'])['reviews'])
    second = keyed(readjson(paths['review_independent'])['stories'])
    pages = keyed(readjson(paths['license_manifest'])['stories'])
    audit = readjson(AUDIT)
    for k,name in [('page_verification','page-text-verification-v2.json'),('review_a','quality-review-a.json'),('review_b','quality-review-b.json')]:
        if inputs[k]['sha256'] != audit['verification_inputs'][name]: raise ValueError('historical input mismatch')
    old = {x['source_sha256']: x for x in audit['candidates']}
    if len(old) != len(audit['candidates']): raise ValueError('duplicate audit hash')
    passed = {k for k,v in first.items() if v.get('decision',v.get('verdict')) == '\u901a\u8fc7'}
    if set(second) != passed or len(passed) != 20: raise ValueError('review coverage mismatch')
    selected = sorted(k for k in passed if second[k]['decision'] == '\u901a\u8fc7')
    if len(selected) != 19: raise ValueError('expected 19 stories')
    tok = Tokenizer.from_file(str(VOCAB))
    if tok.get_vocab_size() != 8192: raise ValueError('unexpected tokenizer')
    bos, eos = tok.token_to_id('<|bos|>'), tok.token_to_id('<|eos|>')
    if bos is None or eos is None: raise ValueError('missing boundary tokens')
    docs, manifest = [], []
    for story in selected:
        v, p, a, b = verification[story], pages[story], first[story], second[story]
        source = (BASE/'raw-v1'/f'{story}.md').read_bytes()
        page = (BASE/'license-audit-v1'/f'{story}.html').read_bytes()
        digest = sha(source)
        if any(x['source_sha256'] != digest for x in (v,p,a,b)) or sha(page) != v['page_sha256'] or sha(page) != p['page_sha256']:
            raise ValueError('hash mismatch ' + story)
        if v['status'] != 'matched_pending_quality_and_rights_review' or not v['text_matches_page'] or not v['allowed_page_license'] or p['source_label'] != 'CC-BY':
            raise ValueError('verification failed ' + story)
        title, paragraphs, attrs = extract(source.decode('utf-8-sig'), v['paragraphs'])
        parser = Page(); parser.feed(page.decode('utf-8'))
        if [compact(x) for x in paragraphs] != [compact(x) for x in parser.paragraphs]: raise ValueError('paragraph mismatch ' + story)
        if parser.licenses != set(v['license_urls']) or parser.licenses != set(p['license_urls']) or len(parser.licenses) != 1 or not parser.licenses <= ALLOWED:
            raise ValueError('license mismatch ' + story)
        if any(attrs[k] != v['attribution'][k] for k in ('Text','Illustration','Translation','License')): raise ValueError('attribution mismatch')
        if p['page_url'] != v['page_url'] or p['source_path'] != v['source_path']: raise ValueError('source mismatch')
        text = '\n\n'.join(paragraphs)
        normalized_sha = sha(normalize(text).encode('utf-8'))
        h = old.get(digest)
        if h is None or h['normalized_body_sha256'] != normalized_sha or h['training_approved'] is not False:
            raise ValueError('historical audit mismatch ' + story)
        split, split_hash = split_for(story)
        tokens = len(tok.encode(text).ids)
        doc = {'story': story, 'title': title, 'text': text, 'paragraphs': paragraphs, 'training_approved': False}
        record = {'story': story, 'title': title, 'split': split, 'split_sha256': split_hash,
                  'source_path': v['source_path'], 'source_sha256': digest, 'page_url': v['page_url'],
                  'page_sha256': v['page_sha256'], 'license_url': v['license_urls'][0], 'attribution': attrs,
                  'review_first': a, 'review_independent': b, 'paragraph_count': len(paragraphs),
                  'body_sha256': sha(text.encode('utf-8')), 'normalized_body_sha256': normalized_sha,
                  'characters_including_paragraph_separators': len(text), 'body_tokens': tokens,
                  'tokens_with_bos_eos': tokens+2, 'within_document_next_token_targets': tokens+1,
                  'historical_comparisons': h['comparisons'], 'training_approved': False}
        docs.append(doc); manifest.append(record)
    pair = overlaps([x['text'] for x in docs])
    for rec, comparison in zip(manifest,pair): rec['candidate_comparisons'] = comparison
    if any(any(c.values()) for c in pair): raise ValueError('candidate overlap requires review')
    totals = {}
    for group in ('train','dev'):
        entries = [x for x in manifest if x['split'] == group]
        totals[group] = {'documents': len(entries), **{k: sum(x[k] for x in entries) for k in ('paragraph_count','characters_including_paragraph_separators','body_tokens','tokens_with_bos_eos','within_document_next_token_targets')}}
        if not entries: raise ValueError('empty split')
    out.mkdir(exist_ok=False)
    def write(name, obj):
        with (out/name).open('x', encoding='utf-8', newline='\n') as f: f.write(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    for group in ('train','dev'):
        with (out/(group+'.jsonl')).open('x', encoding='utf-8', newline='\n') as f:
            for d,m in zip(docs,manifest):
                if m['split'] == group: f.write(json.dumps(d,ensure_ascii=False)+'\n')
    result = {'status':'not_approved_for_training', 'inputs':inputs, 'source_revision':readjson(paths['page_verification'])['source_revision'],
              'page_capture_time_utc':readjson(paths['page_verification'])['page_capture_time_utc'],
              'split_rule':'sha256(UTF8("'+SALT+'" + four-digit story)) interpreted big-endian modulo 10: 0=dev, otherwise train',
              'token_count_rule':'existing tokenizer encodes body with paragraphs separated by two LF; add BOS and EOS, count length minus one within each document; no cross-document targets',
              'tokenizer_boundary_ids': {'bos':bos,'eos':eos}, 'historical_audit_scope':audit['scope'],
              'historical_audit_note':'Recorded results bound by source and normalized body hashes. Training-source hashes are from completed audit; source files not re-read in this build.',
              'totals': totals, 'stories':manifest}
    write('manifest.json',result)
    print(json.dumps({'output':str(out),'totals':totals},ensure_ascii=False))

if __name__ == '__main__': main()
