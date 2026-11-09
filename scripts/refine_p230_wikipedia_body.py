"""Conservatively quarantine mechanical damage in the P230 body candidate."""
import argparse
import collections
import hashlib
import json
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MANIFEST = '18ca23d35ca4e82c96c7b98b4d5dabc2dd04165f4fabc9ce3c15421549d15be1'
STATUS = 'pending_review_not_approved_for_training'
RULES = {
    'empty_or_damaged_parentheses': re.compile(r'[（(]\s*(?:[）)]|[，,；;]|(?:或|or)\s*[）)])|[，,；;]\s*[）)]', re.I),
    'empty_name_label': re.compile(r'(?:英語|英语|英文|學名|学名)\s*[:：]\s*(?=[，,；;）)])'),
    'missing_opening_subject': re.compile(r'^\s*(?:是|為|为)(?:一所|一個|一个|一種|一种|一座|一家|一名|一位)'),
    'isolated_reference': re.compile(r'^(?:[^。！？!?\n]{0,35})(?:此本|该版本|該版本)(?:与|與|为|為|是|的)'),
    'missing_location': re.compile(r'(?:位於|位于)\s*[。！？!?]'),
    'missing_named_person': re.compile(r'由\s*(?:先生|女士)\s*(?:於|于|在)'),
    'unspecified_behaviors': re.compile(r'某[两兩]種行為之間'),
    'repeated_sentence_stop': re.compile(r'。。'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reasons(text):
    return [name for name, pattern in RULES.items() if pattern.search(text)]


def build(candidate, reviews, output, expected_manifest=EXPECTED_MANIFEST, count=40, seed=None):
    candidate, output = Path(candidate), Path(output)
    reviews = [Path(p) for p in reviews]
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or count < 1 or not reviews:
        raise ValueError('Invalid sample count or missing reviews')
    manifest_raw = (candidate / 'manifest.json').read_bytes()
    if digest(manifest_raw) != expected_manifest:
        raise ValueError('Candidate manifest digest mismatch')
    manifest = json.loads(manifest_raw)
    if manifest.get('status') != STATUS or set(manifest.get('files', {})) != {'candidates.jsonl', 'excluded.jsonl'}:
        raise ValueError('Candidate manifest mismatch')
    files = {}
    for name, expected in manifest['files'].items():
        raw = (candidate / name).read_bytes()
        if digest(raw) != expected:
            raise ValueError('Candidate file digest mismatch: ' + name)
        files[name] = raw
    review_rows, review_hashes = [], {}
    for path in reviews:
        if path.name in review_hashes:
            raise ValueError('Duplicate review filename')
        raw = path.read_bytes()
        review_hashes[path.name] = digest(raw)
        review_rows.extend(json.loads(raw))
    if len(review_rows) != 40 or {r['index'] for r in review_rows} != set(range(40)):
        raise ValueError('Expected complete 40-item review')
    reviewed = {(r['page_id'], r['text_sha256']) for r in review_rows}
    old_pages = {r['page_id'] for r in review_rows}
    if len(reviewed) != 40 or len(old_pages) != 40:
        raise ValueError('Duplicate review passages or pages')
    seed = seed if seed is not None else secrets.token_hex(32)
    if not isinstance(seed, str) or not seed:
        raise ValueError('Invalid seed')
    kept, quarantined, seen, found, pages = [], [], set(), set(), {}
    counts = collections.Counter({name: 0 for name in RULES})
    kept_tokens = quarantine_tokens = 0
    for line in files['candidates.jsonl'].splitlines():
        row = json.loads(line)
        text = row['text']
        identity = (row['page_id'], row['char_start'], row['char_end'])
        if (row['status'] != STATUS or digest(text.encode('utf8')) != row['text_sha256']
                or identity in seen or type(row['char_start']) is not int or type(row['char_end']) is not int
                or row['char_start'] < 0 or row['char_end'] - row['char_start'] != len(text)
                or type(row['bpe_tokens']) is not int or row['bpe_tokens'] < 0):
            raise ValueError('Invalid candidate passage')
        seen.add(identity)
        key = (row['page_id'], row['text_sha256'])
        if key in reviewed:
            found.add(key)
        flags = reasons(text)
        if flags:
            counts.update(flags)
            quarantined.append({'reasons': flags, 'candidate': row})
            quarantine_tokens += row['bpe_tokens']
        else:
            kept.append(row)
            kept_tokens += row['bpe_tokens']
            if row['page_id'] not in old_pages:
                rank = digest(('span\0' + seed + '\0' + row['page_id'] + '\0' + str(row['char_start'])).encode())
                if row['page_id'] not in pages or rank < pages[row['page_id']][0]:
                    pages[row['page_id']] = (rank, row)
    if (len(seen) != manifest['counts']['passages'] or found != reviewed
            or kept_tokens + quarantine_tokens != manifest['counts']['bpe_tokens'] or len(pages) < count):
        raise ValueError('Candidate totals, review linkage or sample size mismatch')
    selected = sorted(pages, key=lambda page: digest(('page\0' + seed + '\0' + page).encode()))[:count]
    sample = [pages[page][1] for page in selected]
    def jsonl(rows):
        return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf8')
    output_files = {'candidates.jsonl': jsonl(kept), 'quarantined.jsonl': jsonl(quarantined),
                    'sample.json': (json.dumps(sample, ensure_ascii=False, indent=2) + '\n').encode('utf8')}
    result = {'status': STATUS, 'source_manifest_sha256': digest(manifest_raw),
              'source_files_sha256': manifest['files'], 'review_files_sha256': review_hashes,
              'script_sha256': digest(Path(__file__).read_bytes()),
              'counts': {'input_passages': len(seen), 'retained_passages': len(kept),
                         'quarantined_passages': len(quarantined), 'retained_bpe_tokens': kept_tokens,
                         'quarantined_bpe_tokens': quarantine_tokens, 'rule_matches': dict(counts)},
              'sampling': {'seed': seed, 'algorithm': 'sha256 ranked pages and spans, separate domains',
                           'eligible_pages': len(pages), 'sample_pages': len(sample),
                           'excluded_review_pages': len(old_pages)},
              'files': {name: digest(raw) for name, raw in output_files.items()},
              'offsets': manifest.get('offsets'), 'source_repository': manifest.get('source_repository'),
              'revision': manifest.get('revision'), 'license': manifest.get('license'),
              'limitations': ['Mechanical signals only; rules can miss damage or flag valid text. Rules overlap.',
                              'No factual contradiction, completeness, privacy, recency or license verification.',
                              'Candidate only, not approved for training; source limitations still apply.',
                              'Sample is page-weighted, one retained passage per page; not a passage-weighted quality estimate.',
                              'Prior reviewed pages are excluded from the sample, not from the candidate.']}
    output.mkdir(parents=False, exist_ok=False)
    for name, raw in output_files.items():
        (output / name).write_bytes(raw)
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory name directly under runs')
    parser.add_argument('--seed', help='Optional reproducible sampling seed')
    args = parser.parse_args()
    name = args.output
    if name in ('', '.', '..') or any(c in name for c in '/\\:') or Path(name).name != name:
        raise ValueError('Output must be a single directory name')
    reviews = sorted((ROOT / 'runs').glob('p230-body-review-*.json'))
    result = build(ROOT / 'runs/p230-wikipedia-body-candidate-v3', reviews, ROOT / 'runs' / name, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
