"""Build a local, unapproved candidate from the pinned P98 review records."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'runs/wikimedia-intro-review-p98-v1'
VOCAB = ROOT / 'runs/p28-stories-data/candidate-v3/tokenizer.json'
SALT = 'p99-reviewed-intro-split-20261109-v1'
PIN = {
    'manifest.json': 'a4b95240e505a29e6d8b6621caade5b75c7c6faf3179940ff8b7962ba068a0b2',
    'grade-01.json': '5d71f976aaa15e478a4a2ef71adc56badf06e16b557b408aab0231de966f8bee',
    'grade-02.json': '7b862aaf5e3e5b2c08b5df965960c4b627959f23359ef8df6bc3402f66cafcbd',
    'grade-03.json': '5e761fbeb204881a35ef1404f66a6baa375194584b7f15ff78edcf584e2606c4',
    'grade-04.json': 'f38d3ed023e6c584653f46cf4eb7809c3cc78126e14f8f5f3057b0febb7e6c50',
    'cross-grade-01.json': '1e666ffba115c84f261de1840a91de950217a3bb5db5822028c217a33783de6b',
    'cross-grade-02.json': '9e9e8e214f766e7f05293c2ec73822faecab71ff25938ebf11ad2ab4041ce284',
}
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
    if len(shards) != 4 or len(firsts) != 4 or len(seconds) != 2:
        raise ValueError('Wrong number of review files')
    all_rows, first = {}, {}
    for shard, grade in zip(shards, firsts):
        rows = indexed(shard['samples'])
        grades = indexed(grade['samples'])
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


def build(output):
    from tokenizers import Tokenizer
    output = output.resolve()
    runs = (ROOT / 'runs').resolve()
    if output.parent != runs or output.exists():
        raise ValueError('Output must be a new direct child of runs')
    manifest = read_verified(INPUT / 'manifest.json', PIN['manifest.json'])
    if manifest['status'] != 'pending_not_approved_for_training' or set(manifest['files']) != {f'review-{n:02}.json' for n in range(1, 5)}:
        raise ValueError('Unexpected queue manifest')
    shards = [read_verified(INPUT / f'review-{n:02}.json', manifest['files'][f'review-{n:02}.json']) for n in range(1, 5)]
    first = [read_verified(INPUT / f'grade-{n:02}.json', PIN[f'grade-{n:02}.json']) for n in range(1, 5)]
    second = [read_verified(INPUT / f'cross-grade-{n:02}.json', PIN[f'cross-grade-{n:02}.json']) for n in range(1, 3)]
    selected = verify(shards, first, second)
    if len(selected) != 32:
        raise ValueError('Unexpected selected count')
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
              'input_sha256': {**PIN, **manifest['files']}, 'tokenizer_sha256': VOCAB_SHA,
              'builder_sha256': digest(Path(__file__).read_bytes()), 'source_repository': 'wikimedia/wikipedia',
              'source_revision': manifest['source_revision'], 'source_sha256': manifest['raw_source_sha256'],
              'split': {'salt': SALT, 'rule': 'sha256(utf8(salt + NUL + similarity_component)); first 8 hex digits modulo 5 == 0 is development, otherwise train'},
              'counting': 'Unicode code points in unchanged text; tokenizer content ids; targets = content ids + EOS, with BOS as context and no cross-document targets',
              'counts': counts, 'files': files}
    (output / 'manifest.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf8'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/wikimedia-reviewed-intro-p99-v1')
    build(parser.parse_args().output)
