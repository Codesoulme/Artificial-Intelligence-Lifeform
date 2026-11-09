"""Build a page-grouped, review-only P231 candidate from pinned P230 inputs."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from scripts import prepare_p230_wikipedia_body as p230
from scripts import prepare_p231_wikipedia_context as context

PINS = {
    'refined_manifest': '22f9466980324f0dc22ee93e2a4b8b2428ec2bf65d08fcec9ceec61ab809fa02',
    'first_manifest': 'e5e4e96bfb27ca6af61061a09f04ac3e2aaf685506d03a4202e6be5d23e3c434',
    'source': p230.EXPECTED['source'], 'tokenizer': p230.EXPECTED['tokenizer'],
    'reviews': {
        'p230-body-review-0-9.json': '71869de5d0bc9a259166cca1969bd48841c591ad0e841e62c7c6c686baaa00b7',
        'p230-body-review-10-19.json': '444f335fed23bd5b715cb99ab61be0cf4bf392bf2c0ca7bf692a244d7cc0abd5',
        'p230-body-review-20-29.json': '8bbe04fd18bd11f8d4c6954cec6368d1ceb8ee4d8155199be380666d5a2aa6c7',
        'p230-body-review-30-39.json': '89c7a530ffc010c4d1ba6e83e1b69240377156c2d2f5c61b76d2c6b3e4060a9e',
        'p230-refined-review-0-9.json': '673b0ef53309bcfd87dbb772bbf1d33f7413578c2ccfd5e4b47e69a1112be208',
        'p230-refined-review-10-19.json': 'd63f1ca0ffc84a39f924b10dfc1c1849a1b58655d84365d39768cac5d29b0c2b',
        'p230-refined-review-20-29.json': 'b1f8c964b8f923d3a508c2ebb92b9c86ce213f38707ddb5f1c8df8a2e1908608',
        'p230-refined-review-30-39.json': 'd93936e1ace9feb5997b05a74251b788cd8fbba11122fbe45ec6b71b6e16a943',
    },
}
REFINED = Path('runs/p230-wikipedia-body-refined-v1-20261109')
FIRST = Path('runs/p230-wikipedia-body-sample-20261109')
SOURCE = Path('runs/research/wikimedia-wikipedia-zh/shard-00002.parquet')
TOKENIZER = Path('runs/conversation-data-v5/tokenizer.json')
SEED = 'p231-context-review-20261109-v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_verified(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('Digest mismatch: ' + str(path))
    return raw


def verify_file(path, expected):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Digest mismatch: ' + str(path))


def load_inputs(root, pins):
    refined = json.loads(read_verified(root / REFINED / 'manifest.json', pins['refined_manifest']))
    first = json.loads(read_verified(root / FIRST / 'manifest.json', pins['first_manifest']))
    if (refined.get('status') != context.STATUS or first.get('status') != 'training_side_sample_pending_review'
            or first.get('candidate_manifest_sha256') != refined.get('source_manifest_sha256')
            or first.get('candidate_sha256') != refined.get('source_files_sha256', {}).get('candidates.jsonl')):
        raise ValueError('Manifest provenance mismatch')
    raw = read_verified(root / REFINED / 'candidates.jsonl', refined['files']['candidates.jsonl'])
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != refined['counts']['retained_passages']:
        raise ValueError('Retained count mismatch')
    groups, identities = collections.defaultdict(list), set()
    for row in rows:
        pid = row.get('page_id')
        key = (pid, row.get('char_start'), row.get('char_end'))
        if (not isinstance(pid, str) or not pid.isdecimal() or row.get('status') != context.STATUS
                or key in identities or row.get('text_sha256') != context.digest(row['text'])):
            raise ValueError('Duplicate or invalid retained passage')
        identities.add(key)
        groups[pid].append(row)
    samples = [json.loads(read_verified(root / FIRST / 'sample.json', first['sample_sha256'])),
               json.loads(read_verified(root / REFINED / 'sample.json', refined['files']['sample.json']))]
    rounds = []
    for index, sample in enumerate(samples):
        if len(sample) != 40 or len({r['page_id'] for r in sample}) != 40:
            raise ValueError('Invalid review sample')
        for row in sample:
            if (row.get('status') != context.STATUS or row['text_sha256'] != context.digest(row['text'])
                    or (index == 1 and row not in groups.get(row['page_id'], []))):
                raise ValueError('Sample provenance mismatch')
        prefix = 'p230-body-review-' if index == 0 else 'p230-refined-review-'
        names = sorted(name for name in pins['reviews'] if name.startswith(prefix))
        if len(names) != 4:
            raise ValueError('Missing review files')
        reviews = []
        for name in names:
            reviews.extend(json.loads(read_verified(root / 'runs' / name, pins['reviews'][name])))
        if (len(reviews) != 40 or {r['index'] for r in reviews} != set(range(40))
                or any((r['page_id'], r['text_sha256']) !=
                       (sample[r['index']]['page_id'], sample[r['index']]['text_sha256']) for r in reviews)):
            raise ValueError('Review linkage mismatch')
        rounds.append([r['page_id'] for r in sample])
    if set(rounds[0]) & set(rounds[1]):
        raise ValueError('Review rounds overlap')
    verify_file(root / SOURCE, pins['source'])
    verify_file(root / TOKENIZER, pins['tokenizer'])
    return groups, rounds, refined, first


def construct(groups, source_path, count_tokens):
    pf = pq.ParquetFile(source_path)
    stats = collections.Counter()
    seen, candidates, skipped = set(), [], []
    for group_index in range(pf.num_row_groups):
        ids = pf.read_row_group(group_index, columns=['id']).column(0).to_pylist()
        stats['id_rows_decoded'] += len(ids)
        if not any(pid in groups for pid in ids):
            continue
        table = pf.read_row_group(group_index, columns=['id', 'title', 'url', 'text'])
        stats['full_row_groups_decoded'] += 1
        stats['full_rows_decoded'] += len(ids)
        for i, pid in enumerate(ids):
            if pid not in groups:
                continue
            if pid in seen:
                raise ValueError('Duplicate source page: ' + pid)
            seen.add(pid)
            row = {name: table.column(name)[i].as_py() for name in ('id', 'title', 'url', 'text')}
            # Validate even pages whose title cannot safely be presented to the model.
            context._validated_spans(row, groups[pid])
            if not context._safe_label(row['title']):
                skipped.append({'page_id': pid, 'reason': 'unsafe_title'})
                continue
            candidates.extend(context.build_page_candidates(row, groups[pid], count_tokens))
    if seen != set(groups):
        raise ValueError('Missing source pages: ' + str(len(set(groups) - seen)))
    stats['source_pages_matched'] = len(seen)
    stats['pages_skipped_unsafe_title'] = len(skipped)
    return candidates, skipped, stats


def deduplicate(candidates):
    """Remove exact body duplicates, keeping the first page and position in sorted order."""
    kept, duplicates, bodies = [], [], {}
    for row in sorted(candidates, key=lambda r: (r['page_id'], r['char_start'], r['char_end'])):
        key = row['text']
        if key in bodies:
            first = bodies[key]
            duplicates.append({'page_id': row['page_id'], 'char_start': row['char_start'],
                               'char_end': row['char_end'], 'text_sha256': row['text_sha256'],
                               'kept_page_id': first['page_id'], 'kept_char_start': first['char_start']})
        else:
            bodies[key] = row
            kept.append(row)
    return kept, duplicates


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def build(root, output, pins=PINS, seed=SEED, sample_count=40):
    root, output = Path(root), Path(output)
    if output.parent.resolve() != (root / 'runs').resolve() or output.exists():
        raise ValueError('Output must be a new direct child of runs')
    groups, rounds, refined, first = load_inputs(root, pins)
    tokenizer = Tokenizer.from_file(str(root / TOKENIZER))
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)
    candidates, skipped, stats = construct(groups, root / SOURCE, count_tokens)
    candidates, duplicates = deduplicate(candidates)
    sample = context.sample_candidates(candidates, rounds, sample_count, seed)
    pages = collections.defaultdict(list)
    for row in candidates:
        pages[row['page_id']].append(row)
    page_rows = [{'page_id': pid, 'candidates': rows} for pid, rows in sorted(pages.items())]
    outputs = {
        'candidates.jsonl': b''.join((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8') for r in page_rows),
        'duplicates.jsonl': b''.join((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8') for r in duplicates),
        'skipped.jsonl': b''.join((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8') for r in skipped),
        'sample.json': json_bytes(sample),
    }
    totals = {key: sum(r[key] for r in candidates) for key in
              ('body_bpe_tokens', 'extra_heading_bpe_tokens', 'model_bpe_tokens', 'model_minus_body_bpe_tokens')}
    stats.update({'candidate_pages': len(pages), 'candidate_passages': len(candidates),
                  'exact_body_duplicates_removed': len(duplicates), 'sample_pages': len(sample)})
    manifest = {'status': context.STATUS, 'input_sha256': pins,
                'refined_candidate_sha256': refined['files']['candidates.jsonl'],
                'review_sample_sha256': [first['sample_sha256'], refined['files']['sample.json']],
                'script_sha256': p230.sha(Path(__file__)),
                'context_script_sha256': p230.sha(Path(context.__file__)),
                'source_repository': refined.get('source_repository'), 'revision': refined.get('revision'),
                'license': refined.get('license'), 'offsets': refined.get('offsets'),
                'sampling': {'seed': seed, 'method': 'one hash-ranked passage per hash-ranked page',
                             'excluded_review_pages': 80},
                'counts': dict(sorted(stats.items())), 'token_totals': totals,
                'files': {name: sha(data) for name, data in outputs.items()},
                'limitations': [
                    'Pending review only; not approved or certified for training.',
                    'No guarantee of context, factual accuracy, privacy, recency, attribution or license clearance.',
                    'P230 exclusions and provenance apply; other ancestral sources, development overlap and semantic duplicates remain unknown.',
                    'All source IDs are decoded; full fields are decoded for selected row groups and may include unselected pages, but only candidate pages are constructed or output.',
                    'Exact body duplicates are removed; partial and semantic duplicates are not detected.',
                    'Body totals are candidate token sums, not newly sourced text; titles and sections may repeat across passages.',
                    'Heading token totals count each prefix separately; BPE is not additive, so the model total is measured independently.',
                    'Heading classification is heuristic and may mistake short incomplete lines for headings.',
                    'Sample is page-weighted and excludes both previously reviewed 40-page samples, not a passage-weighted quality estimate.' ]}
    output.mkdir(parents=False, exist_ok=False)
    for name, data in outputs.items():
        (output / name).write_bytes(data)
    (output / 'manifest.json').write_bytes(json_bytes(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory name under runs')
    parser.add_argument('--seed', default=SEED)
    args = parser.parse_args()
    if not args.output or args.output in ('.', '..') or any(c in args.output for c in '/\\:'):
        parser.error('Output must be one directory name')
    print(json.dumps(build(ROOT, ROOT / 'runs' / args.output, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
