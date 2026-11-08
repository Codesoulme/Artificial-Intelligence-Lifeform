"""Build a local, unapproved candidate from the pinned P99 review records."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'runs/wikimedia-intro-review-p99-v1-repro'
PRIOR = ROOT / 'runs/wikimedia-reviewed-intro-p99-v1'
PRIOR_MANIFEST_SHA = 'd09f98f4ca861e5684b0f749dd620b83ceaeedd3444bc9775d35f474bef982b5'
VOCAB = ROOT / 'runs/p28-stories-data/candidate-v3/tokenizer.json'
SALT = 'p99-reviewed-intro-split-20261109-v1'
PIN = {'manifest.json': '96f30c80d7c894798fade0107985de6f59ef7f5ae4ac5e8d3f4b773f825ba327', 'grade-01.json': 'fa9517a58e6af5a7d2ea85b4529714565fc41e8ecf663f1062f2c086a5bdda0a', 'grade-02.json': '38dc00f1f346bbba17c5331ed9e246a6f12ad3e9d66d74b744051345850bf4a2', 'grade-03.json': 'ebd08292fb7b170afaf08ba018325d97908638de5c0259fb875f117a77e3e86d', 'grade-04.json': '5d74fb0dd58491a902d6e2bf7743252f760c6b79f30cb4a5990cb17a504dd255', 'cross-grade-01.json': '43529ba785c47ae711f0c75aa3259bcf8fb1d62bb77e09f855cc6e10fba2bb61', 'cross-grade-02.json': 'f94b23eec965a172d03e347021329156fda06ec4cb8df6360cee0d8d971dc739', 'cross-grade-03.json': '074952016c18701bfec01113e0c6636833a533c5d8d73432607b179520ea5d28', 'cross-grade-04.json': 'dd56fbcb8e2856f02df1aabe9729c91929c61e09daddf89899beab601b506414'}
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
        grades = indexed(grade['结果'] if '结果' in grade else grade['samples'])
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



def load_prior():
    manifest = read_verified(PRIOR / 'manifest.json', PRIOR_MANIFEST_SHA)
    if manifest['status'] != 'candidate_not_approved_for_training' or manifest['split']['salt'] != SALT:
        raise ValueError('Prior candidate status or split mismatch')
    if set(manifest['files']) != {'train.jsonl', 'development.jsonl'}:
        raise ValueError('Unexpected prior files')
    rows = []
    for split in ('train', 'development'):
        data = (PRIOR / f'{split}.jsonl').read_bytes()
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
    if len(selected) != 80:
        raise ValueError('Unexpected selected count')
    prior_manifest, prior_rows = load_prior()
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
              'prior_files_sha256': prior_manifest['files'], 'isolation': {'page_overlap': 0, 'similarity_component_overlap': 0}, 'tokenizer_sha256': VOCAB_SHA,
              'builder_sha256': digest(Path(__file__).read_bytes()), 'source_repository': 'wikimedia/wikipedia',
              'source_revision': manifest['source_revision'], 'source_sha256': manifest['raw_source_sha256'],
              'split': {'salt': SALT, 'rule': 'sha256(utf8(salt + NUL + similarity_component)); first 8 hex digits modulo 5 == 0 is development, otherwise train'},
              'counting': 'Unicode code points in unchanged text; tokenizer content ids; targets = content ids + EOS, with BOS as context and no cross-document targets',
              'counts': counts, 'files': files}
    (output / 'manifest.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf8'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/wikimedia-reviewed-intro-p100-v1')
    build(parser.parse_args().output)
