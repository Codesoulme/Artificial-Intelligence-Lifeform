"""Prepare a small traceable mixture from previously reviewed training-side rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

from scripts.p115_encode_stories import identities

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ('wiki99', 'runs/wikimedia-reviewed-intro-p99-v1', 'd09f98f4ca861e5684b0f749dd620b83ceaeedd3444bc9775d35f474bef982b5'),
    ('wiki100', 'runs/wikimedia-reviewed-intro-p100-v1', 'cbd35121d4a4e8c5c39a17d99822a2466a60926ac21cc09511ed781e5996c1f5'),
    ('wiki101', 'runs/wikimedia-reviewed-intro-p101-v1', '89c28ec1eec48453a766427bbe367f62d4308c20f2d0b331230c5e70513fb4a3'),
    ('voyage', 'runs/p222-wikivoyage-reviewed-sample-20261109', '5547544208cf75bfc0661912d2a799b57e6330c7e5e11a83cf76bb3d7bce2ad0'),
)
VOCAB_SHA = 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(root, tokenizer_path, output):
    root, tokenizer_path, output = Path(root), Path(tokenizer_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if sha(tokenizer_path.read_bytes()) != VOCAB_SHA:
        raise ValueError('Tokenizer digest mismatch')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows, names, titles, counts = [], set(), {}, Counter()
    input_hashes = {}
    for source, directory, expected in SOURCES:
        directory = root/directory
        manifest_raw = (directory/'manifest.json').read_bytes()
        if sha(manifest_raw) != expected:
            raise ValueError('Source manifest changed: '+source)
        manifest = json.loads(manifest_raw)
        filename = 'candidate.jsonl' if source == 'voyage' else 'train.jsonl'
        data = (directory/filename).read_bytes()
        expected_data = manifest['output_sha256'] if source == 'voyage' else manifest['files']['train.jsonl']
        if sha(data) != expected_data:
            raise ValueError('Source content changed: '+source)
        input_hashes[source] = {'manifest_sha256': expected, 'content_sha256': sha(data)}
        for line in data.splitlines():
            row = json.loads(line)
            text, title, page = row['text'], row['title'], row['page_id']
            if not isinstance(text, str) or sha(text.encode('utf8')) != row['text_sha256']:
                raise ValueError('Invalid source text')
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids or len(ids) > 510 or tokenizer.decode(ids, skip_special_tokens=False) != text:
                counts[source+'_length_or_roundtrip_rejected'] += 1
                continue
            identity, group = identities(text)
            if identity in names:
                counts[source+'_exact_normalized_duplicate'] += 1
                continue
            names.add(identity)
            title_key = title.casefold().replace(' ', '')
            titles.setdefault(title_key, set()).add('voyage' if source == 'voyage' else 'wikipedia')
            rows.append({'source_row_1based': len(rows)+1, 'text': text, 'sha256': identity,
                         'group': group, 'source_original_sha256': row['text_sha256'],
                         'origin_source': source, 'origin_page_id': page,
                         'origin_group': row.get('similarity_component', row.get('group')),
                         'origin_title': title, 'origin_url': row.get('source_url', row.get('url')),
                         'origin_revision': row.get('revision_id'), 'content_tokens': len(ids)})
            counts[source+'_records'] += 1
            counts[source+'_content_tokens'] += len(ids)
    # The normalized title check is only a conservative signal; it does not prove isolation.
    common_titles = sorted(title for title, origins in titles.items() if len(origins) > 1)
    if common_titles:
        raise ValueError('Cross-source titles require review before use')
    if not rows:
        raise ValueError('Empty candidate')
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in rows).encode('utf8')
    result = {'status': 'model_reviewed_bounded_pilot_candidate', 'input_hashes': input_hashes,
              'tokenizer_sha256': VOCAB_SHA, 'candidate_sha256': sha(raw),
              'counts': {**counts, 'records': len(rows), 'content_tokens': sum(r['content_tokens'] for r in rows)},
              'test_split_read': False,
              'limitations': ['Small two-domain sample, reviewed by models rather than humans.',
                              'Wikipedia page revision and full rights chain remain unverified.',
                              'Individual passages may be stale, biased or factually incorrect.',
                              'Cross-source title comparison cannot detect translations or paraphrases.',
                              'Sampling with replacement requires accounting for repeated targets.']}
    output.mkdir(parents=False)
    (output/'candidate.jsonl').write_bytes(raw)
    (output/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(ROOT, ROOT/'runs/conversation-data-v5/tokenizer.json', args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
