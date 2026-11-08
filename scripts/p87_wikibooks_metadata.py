"""P87 metadata-only prefix inventory; no text extraction or sampling."""
import bz2
import collections
import hashlib
import json
from pathlib import Path
from xml.parsers import expat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/research/zhwikibooks/pages-official.xml.bz2'
AUDIT = ROOT / 'runs/research/zhwikibooks/train-audit-v1.json'
OUTPUT = ROOT / 'runs/research/zhwikibooks-p87-metadata-v1'
# Fixed title-substring routing. Matches are evidence, not judgments of content.
RULES = (
    ('A', ('语言', '語言', '文学', '文學', '历史', '歷史', '文化', '语文', '語文')),
    ('B', ('生活', '饮食', '飲食', '食谱', '食譜', '社会', '社會')),
    ('C', ('科学', '科學', '生物', '物理', '化学', '化學', '医学', '醫學', '健康')),
    ('D', ('数学', '數學', '计算机', '計算機', '编程', '編程', '工程')),
    ('E', ('法律', '法學', '法学', '政府', '法规', '法規', '判例', '命令註解')),
)
LABELS = {'A': '语言文学及历史文化', 'B': '生活、饮食与社会常识',
          'C': '自然科学与健康', 'D': '数学、计算机与工程',
          'E': '法律、制度与政府文件', 'F': '其他或无法判定'}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def group_hash(title):
    return hashlib.sha256(title.split('/', 1)[0].encode('utf-8')).hexdigest()


def classify(prefix):
    matches = [(category, word) for category, words in RULES for word in words if word in prefix]
    categories = list(dict.fromkeys(category for category, _ in matches))
    return {'category': categories[0] if categories else 'F',
            'basis': [{'category': category, 'title_substring': word} for category, word in matches],
            'category_ambiguity': 'multiple_categories' if len(categories) > 1 else
                                  ('no_title_keyword' if not categories else 'title_only_unverified')}


def read_metadata(path, excluded):
    """Discard all text-element character data in Expat; never build a page body."""
    groups = {}
    counts = collections.Counter()
    parser = expat.ParserCreate(namespace_separator='}')
    stack, page, capture = [], None, []

    def start(name, attrs):
        nonlocal page, capture
        tag = name.rsplit('}', 1)[-1]
        stack.append(tag)
        if tag == 'page':
            page = {'redirect': False}
        elif page is not None and tag == 'redirect' and stack[-2:] == ['page', 'redirect']:
            page['redirect'] = True
        if page is not None and (stack[-2:] in (['page', 'title'], ['page', 'ns']) or
                                 stack[-3:] == ['page', 'revision', 'id']):
            capture = []
            parser.CharacterDataHandler = capture.append

    def end(name):
        nonlocal page
        tag = name.rsplit('}', 1)[-1]
        if page is not None and (stack[-2:] in (['page', 'title'], ['page', 'ns']) or
                                 stack[-3:] == ['page', 'revision', 'id']):
            page[tag if tag != 'id' else 'revision_id'] = ''.join(capture)
            parser.CharacterDataHandler = None
        if tag == 'page':
            counts['all_pages'] += 1
            if page.get('ns') == '0' and not page['redirect']:
                counts['main_nonredirect_pages'] += 1
                title = page.get('title', '')
                if not title or not page.get('revision_id'):
                    counts['missing_title_or_revision'] += 1
                else:
                    prefix = title.split('/', 1)[0]
                    sha = group_hash(title)
                    if int(sha[:8], 16) % 10 == 0:
                        counts['reserved_bucket_pages'] += 1
                    elif sha in excluded or prefix in excluded:
                        counts['old_sample_prefix_pages'] += 1
                    else:
                        counts['eligible_metadata_pages'] += 1
                        if prefix not in groups:
                            groups[prefix] = {'prefix': prefix, 'prefix_sha256': sha,
                                              **classify(prefix), 'pages': [],
                                              'group_ambiguity': 'prefix_only_aliases_and_relationships_unresolved'}
                        groups[prefix]['pages'].append({'title': title, 'revision_id': page['revision_id']})
            page = None
        stack.pop()

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    with bz2.open(path, 'rb') as f:
        while chunk := f.read(65536):
            parser.Parse(chunk, False)
        parser.Parse(b'', True)
    result = []
    for prefix in sorted(groups):
        group = groups[prefix]
        group['pages'].sort(key=lambda x: (x['title'], x['revision_id']))
        group['page_count'] = len(group['pages'])
        result.append(group)
    return counts, result


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if audit['status'] != 'sample_not_approved_for_training' or len(audit['samples']) != 40:
        raise ValueError('Unexpected audit status or sample count')
    if digest(SOURCE) != audit['source_sha256']:
        raise ValueError('Source SHA-256 does not match the audit')
    excluded = set()
    for sample in audit['samples']:
        prefix = sample['title'].split('/', 1)[0]
        sha = group_hash(prefix)
        if sha != sample['work_sha256']:
            raise ValueError('Sample prefix hash mismatch')
        excluded.update((prefix, sha))
    counts, groups = read_metadata(SOURCE, excluded)
    counts['eligible_prefix_groups'] = len(groups)
    strata = {c: {'groups': sum(g['category'] == c for g in groups),
                  'pages': sum(g['page_count'] for g in groups if g['category'] == c)} for c in LABELS}
    payload = {'schema': 'p87-title-metadata-v1', 'source_sha256': audit['source_sha256'],
               'audit_sha256': digest(AUDIT), 'script_sha256': digest(Path(__file__)),
               'old_sample_distinct_prefixes': len(excluded) // 2, 'counts': dict(counts),
               'strata': strata, 'labels': LABELS, 'groups': groups}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    path = OUTPUT / 'inventory.json'
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'inventory_sha256': digest(path), 'counts': dict(counts), 'strata': strata}, ensure_ascii=False))


if __name__ == '__main__':
    main()
