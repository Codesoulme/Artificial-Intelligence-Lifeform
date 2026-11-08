"""Build a local, unapproved candidate from the pinned P101 review records."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'runs/wikimedia-intro-review-p101-v1'
PRIOR = ROOT / 'runs/wikimedia-reviewed-intro-p99-v1'
PRIOR_MANIFEST_SHA = 'd09f98f4ca861e5684b0f749dd620b83ceaeedd3444bc9775d35f474bef982b5'
VOCAB = ROOT / 'runs/p28-stories-data/candidate-v3/tokenizer.json'
SALT = 'p99-reviewed-intro-split-20261109-v1'
PIN = {'manifest.json': 'aa0068ddf49a55ad6fb7565e13b3593bde3e1818c662a4c9757463b3dbc30540', 'grade-01.json': '7dd713c9ae6c9add1ca85c024de330c47c44d9610f092cfa823e85e1483e41b5', 'grade-02.json': 'cb46acb20436943dff9d843d77c9da69569d2e5f7787578ca05dcdfc1532941b', 'grade-03.json': '9ad4b7c21ba1856980b68e98809db0bf201c9055b484e664e23a8e5c2e99e615', 'grade-04.json': '3c3fe8fdcfb9f268c36d96710a7c022c83432b5e4c3aa577c446a5a2e4eb7935', 'cross-grade-01.json': '8358a2fa759a29ddaa5eccdca641131f432e06e07a7032efe4496be0c77b8c35', 'cross-grade-02.json': '6bad529afde2203129eb3a1bb7183765ab58e19f6568cea59b1fcc9563827df6', 'cross-grade-03.json': '3da9603eb2a20448d1aec02e72a946b2daa64bb68c4794b476cf1df242d9b13b', 'cross-grade-04.json': '0ab66df3e18f4246dfba77bdfb73ff21be8b1c57b4a3ff12b529f831fe52176d'}
VOCAB_SHA = 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return json.loads(data)


def key(row):
    values = tuple(row.get(k) for k in ('page_id', 'group', 'text_sha256'))
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError('Missing review identity')
    return values


def decision(row):
    fields = [row[k] for k in ('decision', 'judgment', '判定') if k in row]
    if len(fields) != 1 or fields[0] not in ('通过', '存疑', '排除'):
        raise ValueError('Invalid or ambiguous decision')
    return fields[0]


def indexed(rows):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError('Duplicate review identity')
        result[k] = row
    return result


def verify(shards, firsts, seconds, expected_shard_size=50):
    if len(shards) != 4 or len(firsts) != 4 or len(seconds) != 4:
        raise ValueError('Wrong number of review files')
    all_rows, first = {}, {}
    for shard, grade in zip(shards, firsts):
        rows = indexed(shard['samples'])
        grades = indexed(grade['结果'] if '结果' in grade else grade['记录'] if '记录' in grade else grade['samples'])
        if len(rows) != expected_shard_size or rows.keys() != grades.keys():
            raise ValueError('First review coverage mismatch')
        if all_rows.keys() & rows.keys():
            raise ValueError('Duplicate queue entry')
        for k, row in rows.items():
            if digest(row['text'].encode('utf8')) != row['text_sha256']:
                raise ValueError('Text hash mismatch')
            if not all(row.get(x) for x in ('similarity_component', 'source_url', 'title')):
                raise ValueError('Missing provenance')
            decision(grades[k])
        all_rows.update(rows)
        first.update(grades)
    if len({row['similarity_component'] for row in all_rows.values()}) != len(all_rows):
        raise ValueError('Duplicate queue component')
    if len({k[0] for k in all_rows}) != len(all_rows):
        raise ValueError('Duplicate page')
    cross = {}
    for item in seconds:
        part = indexed(item['samples'])
        if cross.keys() & part.keys():
            raise ValueError('Duplicate second review')
        for v in part.values():
            decision(v)
        cross.update(part)
    passed = {k for k, v in first.items() if decision(v) == '通过'}
    if cross.keys() != passed:
        raise ValueError('Second review coverage mismatch')
    return [(all_rows[k], first[k], cross[k]) for k in sorted(passed) if decision(cross[k]) == '通过']


def partition(component):
    value = digest((SALT + '\0' + component).encode('utf8'))
    return 'development' if int(value[:8], 16) % 5 == 0 else 'train'



def load_prior(prior, prior_manifest_sha):
    manifest = read_verified(prior / 'manifest.json', prior_manifest_sha)
    if manifest['status'] != 'candidate_not_approved_for_training' or manifest['split']['salt'] != SALT:
        raise ValueError('Prior candidate status or split mismatch')
    if set(manifest['files']) != {'train.jsonl', 'development.jsonl'}:
        raise ValueError('Unexpected prior files')
    rows = []
    for split in ('train', 'development'):
        data = (prior / f'{split}.jsonl').read_bytes()
        if digest(data) != manifest['files'][f'{split}.jsonl']:
            raise ValueError('Prior file SHA-256 mismatch')
        part = [json.loads(line) for line in data.splitlines()]
        if len(part) != manifest['counts'][split]['documents']:
            raise ValueError('Prior count mismatch')
        for row in part:
            if digest(row['text'].encode('utf8')) != row['text_sha256'] or partition(row['similarity_component']) != split:
                raise ValueError('Prior text or split mismatch')
        rows.extend(part)
    if len({r['page_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prior page')
    return manifest, rows


def verify_isolation(selected, prior):
    for field in ('page_id', 'similarity_component'):
        if {r[field] for r, _, _ in selected} & {r[field] for r in prior}:
            raise ValueError(f'Prior candidate overlap: {field}')


def build(output):
    from tokenizers import Tokenizer
    output = output.resolve()
    runs = (ROOT / 'runs').resolve()
    if output.parent != runs or output.exists():
        raise ValueError('Output must be a new direct child of runs')
    manifest = read_verified(INPUT / 'manifest.json', PIN['manifest.json'])
    if manifest['status'] != 'pending_not_approved_for_training' or set(manifest['files']) != {f'review-{n:02}.json' for n in range(1, 5)}:
        raise ValueError('Unexpected queue manifest')
    if manifest['counts']['selected'] != 400 or manifest['counts']['per_shard'] != 100 or manifest['counts']['shards'] != 4:
        raise ValueError('Unexpected manifest counts')
    shards = [read_verified(INPUT / f'review-{n:02}.json', manifest['files'][f'review-{n:02}.json']) for n in range(1, 5)]
    for shard in shards:
        if (shard['status'] != manifest['status'] or shard['source_sha256'] != manifest['raw_source_sha256']
                or shard['train_sha256'] != manifest['train_v3_sha256'] or shard['screened_sha256'] != manifest['p77_screened_sha256']):
            raise ValueError('Shard provenance mismatch')
    first = [read_verified(INPUT / f'grade-{n:02}.json', PIN[f'grade-{n:02}.json']) for n in range(1, 5)]
    second = [read_verified(INPUT / f'cross-grade-{n:02}.json', PIN[f'cross-grade-{n:02}.json']) for n in range(1, 5)]
    selected = verify(shards, first, second, expected_shard_size=100)
    if len(selected) != 99:
        raise ValueError('Unexpected selected count')
    prior_manifest, prior_rows = load_prior(PRIOR, PRIOR_MANIFEST_SHA)
    prior2_manifest_sha = 'cbd35121d4a4e8c5c39a17d99822a2466a60926ac21cc09511ed781e5996c1f5'
    prior2_manifest, prior2_rows = load_prior(ROOT / 'runs/wikimedia-reviewed-intro-p100-v1', prior2_manifest_sha)
    verify_isolation(selected, prior2_rows)
    if prior2_manifest['source_revision'] != manifest['source_revision'] or prior2_manifest['source_sha256'] != manifest['raw_source_sha256']:
        raise ValueError('Second prior source mismatch')
    if prior_manifest['source_revision'] != manifest['source_revision'] or prior_manifest['source_sha256'] != manifest['raw_source_sha256']:
        raise ValueError('Prior source mismatch')
    verify_isolation(selected, prior_rows)
    if digest(VOCAB.read_bytes()) != VOCAB_SHA:
        raise ValueError('Vocabulary SHA-256 mismatch')
    tok = Tokenizer.from_file(str(VOCAB))
    if tok.token_to_id('<|bos|>') is None or tok.token_to_id('<|eos|>') is None:
        raise ValueError('Missing document boundary tokens')
    results = {'train': [], 'development': []}
    for row, a, b in selected:
        content_tokens = len(tok.encode(row['text']).ids)
        item = {k: row[k] for k in ('page_id', 'group', 'similarity_component', 'text_sha256', 'source_url', 'title', 'text')}
        item.update(source_repository='wikimedia/wikipedia', source_revision=manifest['source_revision'], source_sha256=manifest['raw_source_sha256'],
                    first_review=a, second_review=b, rights_status='unverified', fact_status='unverified', training_approval='not_approved',
                    content_characters=len(row['text']), content_tokens=content_tokens, document_targets=content_tokens + 1)
        results[partition(row['similarity_component'])].append(item)
    if not all(results.values()):
        raise ValueError('Empty partition')
    output.mkdir(exist_ok=False)
    files = {}
    counts = {}
    for split, items in results.items():
        data = ''.join(json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for x in items).encode('utf8')
        (output / f'{split}.jsonl').write_bytes(data)
        files[f'{split}.jsonl'] = digest(data)
        counts[split] = {'documents': len(items), 'components': len({x['similarity_component'] for x in items}),
                         'content_characters': sum(x['content_characters'] for x in items),
                         'content_tokens': sum(x['content_tokens'] for x in items),
                         'document_targets': sum(x['document_targets'] for x in items)}
    result = {'status': 'candidate_not_approved_for_training', 'fact_and_rights': 'unverified',
              'input_sha256': {**PIN, **manifest['files']}, 'prior_manifest_sha256': PRIOR_MANIFEST_SHA,
              'prior_files_sha256': prior_manifest['files'], 'second_prior_manifest_sha256': prior2_manifest_sha, 'second_prior_files_sha256': prior2_manifest['files'], 'isolation': {'page_overlap': 0, 'similarity_component_overlap': 0}, 'tokenizer_sha256': VOCAB_SHA,
              'builder_sha256': digest(Path(__file__).read_bytes()), 'source_repository': 'wikimedia/wikipedia',
              'source_revision': manifest['source_revision'], 'source_sha256': manifest['raw_source_sha256'],
              'split': {'salt': SALT, 'rule': 'sha256(utf8(salt + NUL + similarity_component)); first 8 hex digits modulo 5 == 0 is development, otherwise train'},
              'counting': 'Unicode code points in unchanged text; tokenizer content ids; targets = content ids + EOS, with BOS as context and no cross-document targets',
              'counts': counts, 'files': files}
    (output / 'manifest.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf8'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/wikimedia-reviewed-intro-p101-v1')
    build(parser.parse_args().output)
