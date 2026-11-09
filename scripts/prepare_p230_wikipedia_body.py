"""Extract attributed Wikipedia body passages for review, not training approval."""
import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / 'runs/wikimedia-intro-screening-p77-v2/screened.jsonl'
MANIFEST = SCREEN.with_name('manifest.json')
SOURCE = ROOT / 'runs/research/wikimedia-wikipedia-zh/shard-00002.parquet'
TOKENIZER = ROOT / 'runs/conversation-data-v5/tokenizer.json'
EXPECTED = {
    'screen': 'f2a30945dab18b728666bd151a179b720f2ae3a1464a337548a384cdba01879f',
    'manifest': 'f48838670ecc5435ceadb86e7d3241bee40b6bc3a7768a7bc00d616b3cd88d04',
    'source': 'ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f',
    'tokenizer': 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20',
}
EXCLUSIONS = {
    'p99_train': ('runs/wikimedia-reviewed-intro-p99-v1', 'd09f98f4ca861e5684b0f749dd620b83ceaeedd3444bc9775d35f474bef982b5', 'train.jsonl'),
    'p100_train': ('runs/wikimedia-reviewed-intro-p100-v1', 'cbd35121d4a4e8c5c39a17d99822a2466a60926ac21cc09511ed781e5996c1f5', 'train.jsonl'),
    'p101_train': ('runs/wikimedia-reviewed-intro-p101-v1', '89c28ec1eec48453a766427bbe367f62d4308c20f2d0b331230c5e70513fb4a3', 'train.jsonl'),
    'p223_reviewed': ('runs/p223-reviewed-mixture-20261109', '1aaf6e471d0786da5679aecb959b8a5dd5fca333694362a4362a9b5e94fdeb64', 'candidate.jsonl'),
    'p223_isolated': ('runs/p223-mixture-isolated-20261109', 'ed62af729a8d7404077d243790ec01c5cbac2f65bc281f3b8b5f16b2ea4b60a8', 'candidate.jsonl'),
}
QUEUES = {
    'p98_queue': ('runs/wikimedia-intro-review-p98-v1', 'a4b95240e505a29e6d8b6621caade5b75c7c6faf3179940ff8b7962ba068a0b2', 200),
    'p99_queue': ('runs/wikimedia-intro-review-p99-v1-repro', '96f30c80d7c894798fade0107985de6f59ef7f5ae4ac5e8d3f4b773f825ba327', 400),
    'p101_queue': ('runs/wikimedia-intro-review-p101-v1', 'aa0068ddf49a55ad6fb7565e13b3593bde3e1818c662a4c9757463b3dbc30540', 400),
}
CANDIDATE_LINKS = (
    ('p99_train', 'p98_queue', 'scripts/build_p99_reviewed_intro_candidate.py', ()),
    ('p100_train', 'p99_queue', 'scripts/build_p100_reviewed_intro_candidate.py', ('p99_train',)),
    ('p101_train', 'p101_queue', 'scripts/build_p101_reviewed_intro_candidate.py', ('p99_train', 'p100_train')),
)
MAX_CHARS = 6000
END_SECTION = re.compile(r'^(?:参考文献|參考文獻|参考资料|參考資料|注释|註釋|注解|脚注|腳註|外部链接|外部連結|相關條目|相关条目|延伸阅读|延伸閱讀|參見|参见|参考来源|參考來源)\s*$')
NOISE = re.compile(r'https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)|(?:身份证|身份證|住址|家庭住址|手机号|手機號|病史|性侵|强奸|強姦|未成年|自杀|自殺|自残|自殘)|\{\{|\}\}|\[\[|\]\]|<[^>]+>|\\(?:frac|sqrt|begin|end|displaystyle)|\$|\ufffd|(?:ISBN|DOI)\b|\[[0-9]+(?:[ ,，-]+[0-9]+)*\]', re.I)
DEPENDENT = re.compile(r'^(?:此外|另外|因此|但是|然而|同时|同時|其中|这些|這些|上述|例如|而且|并且|並且|该|該|这一|這一|他|她|它|其|亦|也|又|并|並|但|而|或|以及|包括|以及|这|這)[，,、 ]')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized(text):
    return re.sub(r'[^\w]+', '', unicodedata.normalize('NFKC', text).casefold())


def verify_file(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return actual


def load_selection(screen=SCREEN, manifest=MANIFEST, expected=EXPECTED):
    verify_file(manifest, expected['manifest'])
    info = json.loads(manifest.read_text(encoding='utf-8'))
    verify_file(screen, expected['screen'])
    if info['files']['screened.jsonl'] != expected['screen'] or info['status'] != 'mechanical_screen_only_not_approved':
        raise ValueError('Screen manifest mismatch')
    selected = {}
    count = 0
    with screen.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            count += 1
            if type(row['prior_review_group']) is not bool:
                raise ValueError('Invalid prior review flag')
            if row['prior_review_group'] is not False:
                continue
            pid = row['page_id']
            group = text_sha(normalized(row['text']))
            url = urlparse(row['url'])
            if (not isinstance(pid, str) or not pid.isdecimal() or pid in selected
                    or not row['title'] or row['group'] != group or int(group[:8], 16) % 20 == 0
                    or url.scheme != 'https' or url.netloc != 'zh.wikipedia.org'
                    or not url.path.startswith('/wiki/') or url.query or url.fragment):
                raise ValueError('Invalid or duplicate selected page')
            selected[pid] = row
    if count != info['counts']['screened'] or len(selected) != info['eligible_counts']['screened']:
        raise ValueError('Screen count mismatch')
    return selected


def load_exclusions(root=ROOT, specifications=EXCLUSIONS):
    excluded, provenance = {}, {}
    for label, (directory, expected, filename) in specifications.items():
        base = root / directory
        verify_file(base / 'manifest.json', expected)
        manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
        digest = manifest['files'][filename] if filename == 'train.jsonl' else manifest['candidate_sha256']
        verify_file(base / filename, digest)
        ids = set()
        with (base / filename).open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if filename == 'train.jsonl':
                    pid = row['page_id']
                elif row['origin_source'] in ('wiki99', 'wiki100', 'wiki101'):
                    pid = row['origin_page_id']
                else:
                    continue
                if not isinstance(pid, str) or not pid.isdecimal() or pid in ids:
                    raise ValueError('Invalid exclusion page in ' + label)
                ids.add(pid)
        for pid in ids:
            excluded.setdefault(pid, set()).add(label)
        provenance[label] = {'manifest_sha256': expected, 'file_sha256': digest, 'page_count': len(ids)}
    return excluded, provenance


def load_queues(selected, root=ROOT, specifications=QUEUES):
    queues, provenance, manifests = {}, {}, {}
    for label, (directory, digest, count) in specifications.items():
        base = root / directory
        verify_file(base / 'manifest.json', digest)
        manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
        names = {f'review-{n:02}.json' for n in range(1, 5)}
        if (manifest['status'] != 'pending_not_approved_for_training'
                or manifest['raw_source_sha256'] != EXPECTED['source']
                or manifest['p77_manifest_sha256'] != EXPECTED['manifest']
                or manifest['p77_screened_sha256'] != EXPECTED['screen']
                or set(manifest['files']) != names
                or manifest['counts']['selected'] != count
                or manifest['counts']['shards'] != 4
                or manifest['counts']['per_shard'] * 4 != count):
            raise ValueError('Queue manifest mismatch: ' + label)
        pages = set()
        for name in sorted(names):
            verify_file(base / name, manifest['files'][name])
            shard = json.loads((base / name).read_text(encoding='utf-8'))
            if (shard['status'] != manifest['status']
                    or shard['source_sha256'] != manifest['raw_source_sha256']
                    or shard['train_sha256'] != manifest['train_v3_sha256']
                    or shard['screened_sha256'] != manifest['p77_screened_sha256']
                    or len(shard['samples']) != manifest['counts']['per_shard']):
                raise ValueError('Queue shard mismatch: ' + label + '/' + name)
            for row in shard['samples']:
                pid = row['page_id']
                source = selected.get(pid)
                if (pid in pages or source is None
                        or any(row[k] != source[v] for k, v in (('group', 'group'), ('title', 'title'), ('source_url', 'url'), ('text', 'text')))
                        or row['text_sha256'] != text_sha(source['text'])):
                    raise ValueError('Queue page mismatch: ' + label)
                pages.add(pid)
        if len(pages) != count or any(pages & other for other in queues.values()):
            raise ValueError('Queue coverage or overlap mismatch: ' + label)
        queues[label], manifests[label] = pages, manifest
        provenance[label] = {'manifest_sha256': digest, 'files': manifest['files'], 'page_count': len(pages)}
    for child, parent in (('p99_queue', 'p98_queue'), ('p101_queue', 'p98_queue'), ('p101_queue', 'p99_queue')):
        if child in manifests and parent in manifests:
            prefix = parent.removesuffix('_queue')
            if (manifests[child].get(prefix + '_manifest_sha256') != specifications[parent][1]
                    or manifests[child].get(prefix + '_files') != manifests[parent]['files']):
                raise ValueError('Queue inheritance mismatch: ' + child)
    return queues, manifests, provenance


def verify_candidate_links(queues, queue_manifests, root=ROOT):
    candidate_manifests = {}
    for label, queue, builder, prior in CANDIDATE_LINKS:
        directory, expected, filename = EXCLUSIONS[label]
        base = root / directory
        verify_file(base / 'manifest.json', expected)
        manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
        verify_file(root / builder, manifest['builder_sha256'])
        source = queue_manifests[queue]
        if (manifest['status'] != 'candidate_not_approved_for_training'
                or manifest['source_sha256'] != EXPECTED['source']
                or manifest['input_sha256']['manifest.json'] != QUEUES[queue][1]
                or any(manifest['input_sha256'].get(k) != v for k, v in source['files'].items())):
            raise ValueError('Candidate queue provenance mismatch: ' + label)
        for i, prior_label in enumerate(prior):
            key = 'prior' if i == 0 else 'second_prior'
            if (manifest[key + '_manifest_sha256'] != EXCLUSIONS[prior_label][1]
                    or manifest[key + '_files_sha256'] != candidate_manifests[prior_label]['files']):
                raise ValueError('Candidate inheritance mismatch: ' + label)
        verify_file(base / filename, manifest['files'][filename])
        with (base / filename).open(encoding='utf-8') as f:
            ids = [json.loads(line)['page_id'] for line in f]
        if (len(ids) != manifest['counts']['train']['documents'] or len(set(ids)) != len(ids)
                or not set(ids) <= queues[queue]
                or len(ids) + manifest['counts']['development']['documents'] > len(queues[queue])):
            raise ValueError('Candidate train coverage mismatch: ' + label)
        candidate_manifests[label] = manifest
    return {'verified': 'Pinned candidate manifests reference the corresponding complete queue, pinned builders and prior manifests; verified training IDs belong to their queues.',
            'unknown': 'Development output bytes and actual page membership were not inspected or reconstructed; their queue coverage relies on the pinned builder and manifest provenance. Other development sources and semantic overlap remain unknown.'}


def paragraphs(text):
    """Yield nonempty original lines with character offsets, including headings."""
    for match in re.finditer(r'[^\r\n]+', text):
        raw = match.group()
        start = match.start() + len(raw) - len(raw.lstrip())
        end = match.start() + len(raw.rstrip())
        if start < end:
            yield start, end, text[start:end]


def rejection(text):
    if len(text) > MAX_CHARS:
        return 'oversize'
    if len(text) < 35 or not re.search(r'[。！？!?][”’」』）)]?$', text):
        return 'short_heading_or_incomplete'
    if NOISE.search(text) or re.match(r'^(?:[*#•●▪]|\d+[.、)]|[一二三四五六七八九十]+、)', text):
        return 'format_or_sensitive'
    if sum('\u4e00' <= c <= '\u9fff' for c in text) / len(text) < .35:
        return 'low_chinese_ratio'
    if any(text.count(a) != text.count(b) for a, b in [('（', '）'), ('(', ')'), ('《', '》'), ('「', '」'), ('“', '”')]):
        return 'unbalanced'
    return None


def extract(text, stats):
    """Yield continuous source slices; rejected lines always break a passage."""
    current = []
    def flush():
        nonlocal current
        result = current
        current = []
        return result
    for start, end, para in paragraphs(text):
        if END_SECTION.fullmatch(para):
            if current:
                yield flush()
            stats['terminal_sections'] += 1
            break
        reason = rejection(para)
        if reason:
            if current:
                yield flush()
            stats['rejected_' + reason] += 1
            continue
        if current and end - current[0][0] > MAX_CHARS:
            yield flush()
        if not current and DEPENDENT.match(para):
            stats['rejected_dependent_opening'] += 1
            continue
        current.append((start, end))
    if current:
        yield flush()


def output_path(name):
    if not name or name in ('.', '..') or any(c in name for c in '/\\:') or Path(name).name != name:
        raise ValueError('Output must be a single new directory name under runs')
    return ROOT / 'runs' / name


def build(out):
    if out.parent.resolve() != (ROOT / 'runs').resolve():
        raise ValueError('Output must be a direct child of runs')
    if out.exists():
        raise FileExistsError(out)
    selected = load_selection()
    excluded, exclusion_provenance = load_exclusions()
    queues, queue_manifests, queue_provenance = load_queues(selected)
    candidate_coverage = verify_candidate_links(queues, queue_manifests)
    for label, pages in queues.items():
        for pid in pages:
            excluded.setdefault(pid, set()).add(label)
    excluded_rows = [{'page_id': pid, 'reasons': sorted(excluded[pid])} for pid in sorted(selected.keys() & excluded.keys())]
    for item in excluded_rows:
        del selected[item['page_id']]
    verify_file(SOURCE, EXPECTED['source'])
    verify_file(TOKENIZER, EXPECTED['tokenizer'])
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    pf = pq.ParquetFile(SOURCE)
    stats = collections.Counter()
    seen = set()
    out.mkdir(parents=False, exist_ok=False)
    (out / 'excluded.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in excluded_rows), encoding='utf-8', newline='\n')
    with (out / 'candidates.jsonl').open('x', encoding='utf-8', newline='\n') as dest:
        for group_index in range(pf.num_row_groups):
            ids = pf.read_row_group(group_index, columns=['id']).column(0).to_pylist()
            stats['id_rows_decoded'] += len(ids)
            if not any(pid in selected for pid in ids):
                continue
            table = pf.read_row_group(group_index, columns=['id', 'title', 'url', 'text'])
            stats['full_rows_decoded'] += len(ids)
            stats['full_row_groups_decoded'] += 1
            for i, pid in enumerate(ids):
                if pid not in selected:
                    continue
                if pid in seen:
                    raise ValueError('Duplicate source page: ' + pid)
                seen.add(pid)
                row = {name: table.column(name)[i].as_py() for name in ('id', 'title', 'url', 'text')}
                source = selected[pid]
                if any(row[k] != source[k] for k in ('title', 'url')) or not isinstance(row['text'], str):
                    raise ValueError('Source identity mismatch: ' + pid)
                intro = re.split(r'\n\s*\n', unicodedata.normalize('NFKC', row['text']).strip(), maxsplit=1)[0].strip()
                intro = re.sub(r'\s+', ' ', intro)
                if intro != source['text']:
                    raise ValueError('Source introduction mismatch: ' + pid)
                stats['selected_pages_matched'] += 1
                full_hash = text_sha(row['text'])
                for spans in extract(row['text'], stats):
                    start, end = spans[0][0], spans[-1][1]
                    passage = row['text'][start:end]
                    item = {'status': 'pending_review_not_approved_for_training', 'page_id': pid,
                            'title': row['title'], 'url': row['url'], 'source_text_sha256': full_hash,
                            'text': passage, 'text_sha256': text_sha(passage), 'char_start': start,
                            'char_end': end, 'paragraphs': [{'char_start': a, 'char_end': b} for a, b in spans],
                            'bpe_tokens': len(tokenizer.encode(passage, add_special_tokens=False).ids)}
                    dest.write(json.dumps(item, ensure_ascii=False) + '\n')
                    stats['passages'] += 1
                    stats['bpe_tokens'] += item['bpe_tokens']
    if seen != set(selected):
        raise ValueError(f'Missing selected pages: {len(set(selected) - seen)}')
    stats['excluded_selected_pages'] = len(excluded_rows)
    manifest = {'status': 'pending_review_not_approved_for_training', 'input_sha256': EXPECTED,
                'exclusion_sources': exclusion_provenance, 'review_queues': queue_provenance,
                'candidate_coverage': candidate_coverage,
                'script_sha256': sha(Path(__file__)), 'source_repository': 'wikimedia/wikipedia',
                'revision': 'b04c8d1ceb2f5cd4588862100d08de323dccfbaa',
                'license': 'CC-BY-SA-3.0 and GFDL per dataset card; retain attribution and review license',
                'offsets': 'Python Unicode code points in original source text; end exclusive',
                'token_count': 'content BPE tokens without special tokens', 'max_chars': MAX_CHARS,
                'counts': dict(sorted(stats.items())), 'files': {name: sha(out / name) for name in ('candidates.jsonl', 'excluded.jsonl')},
                'limitations': ['Only P77 train-side screened pages with prior_review_group=false are selected; this does not prove absence of all evaluation overlap.',
                    'All P98/P99/P101 review queue page IDs are excluded; coverage of the P99/P100/P102 development outputs relies on the pinned candidate builders and manifests, not inspection of those output files.',
                    'P223 Wikipedia pages and all pages in the three reviewed train lists are excluded by page ID; other training histories and near duplicates are not checked.',
                    'Parquet ID columns are decoded for all rows; all columns are decoded for every row in selected row groups, including unselected pages.',
                    'Mechanical filters are incomplete; passages need privacy, quality, continuity, attribution and license review before use.',
                    'Oversize or incomplete paragraphs are discarded whole; no passage crosses a discarded line.']}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='p230-wikipedia-body-candidate-v3')
    args = parser.parse_args()
    print(json.dumps(build(output_path(args.output)), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
