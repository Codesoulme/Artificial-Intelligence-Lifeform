"""Extract reviewed training-side pairs for local research, without approving them."""
import argparse
import hashlib
import json
from pathlib import Path

from spora.paired_order import PairedOrder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'runs/p104-paired-candidate-v1'
RUNS = ROOT / 'runs'
PINS = {
    'manifest.json': '8bf22705d4830bcd02d66d1ab2a1670a9261fe43f0d43ff556ee850869abea46',
    'context.jsonl': '7b8cbf3c63f92deafb5553b17b61ad52be87e0933f7ea0c271bb1c5b090d2761',
    'last-user.jsonl': 'd06c079b7f718ec0cd2a0eca0739161592bf3ba6412d1c64926d616f3d630934',
    'lineage.jsonl': 'c4572617618a8a63fd7e3b8d02ba789cd606685c07b67236ecc947d8b1c09b6a',
    'p107-cross-review-queue.json': 'b5c34f1b2b181b3039d4885aee4b4e9bc98243b25e716e53ea5feba62107310b',
    'p107-cross-review-1.json': '8857e9962df184764c843dcd710ff078c795bfeb01f02910fb6bb91d4d98ce66',
    'p107-cross-review-33.json': '07e9b1cb8b74029c0d7daec02f60f4cbe5ef4b887fb62e50696bd63c796dd7af',
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'Changed input: {path.name}')
    return data


def select(queue, reviews, context, control, lineage):
    def keyed(rows):
        result = {}
        for row in rows:
            key = row.get('pair_id')
            if not isinstance(key, str) or not key or key in result or not row.get('group'):
                raise ValueError('Invalid or duplicate identity')
            result[key] = row
        return result
    targets = keyed(queue)
    result = keyed(reviews)
    if result.keys() != targets.keys():
        raise ValueError('Review coverage mismatch')
    for key, row in result.items():
        if row['group'] != targets[key]['group'] or row.get('判定') not in ('通过', '存疑', '排除'):
            raise ValueError('Review identity or decision mismatch')
    a, b, meta = keyed(context), keyed(control), keyed(lineage)
    if a.keys() != b.keys() or a.keys() != meta.keys() or not targets.keys() <= a.keys():
        raise ValueError('Source coverage mismatch')
    PairedOrder(context, control, seed=0, epochs=1, length=511)
    if any(a[k]['group'] != meta[k]['group'] or a[k]['group'] != v['group'] for k, v in targets.items()):
        raise ValueError('Source group mismatch')
    selected = sorted(k for k, v in result.items() if v['判定'] == '通过')
    if len({a[k]['group'] for k in selected}) != len(selected):
        raise ValueError('Multiple selected pairs for one group')
    return [a[k] for k in selected], [b[k] for k in selected], [meta[k] for k in selected]


def build(output):
    output = Path(output).resolve()
    if output.parent != RUNS.resolve() or output.exists():
        raise ValueError('Output must be a new direct child of runs')
    manifest = json.loads(read(DATA / 'manifest.json', PINS['manifest.json']))
    def jsonl(name):
        return [json.loads(line) for line in read(DATA / name, PINS[name]).splitlines()]
    context, control, lineage = (jsonl(x) for x in ('context.jsonl', 'last-user.jsonl', 'lineage.jsonl'))
    queue = json.loads(read(RUNS / 'p107-cross-review-queue.json', PINS['p107-cross-review-queue.json']))['records']
    reviews = []
    for name in ('p107-cross-review-1.json', 'p107-cross-review-33.json'):
        value = json.loads(read(RUNS / name, PINS[name]))
        reviews.extend(value if isinstance(value, list) else value['records'])
    if len(queue) != 63:
        raise ValueError('Unexpected review queue size')
    a, b, meta = select(queue, reviews, context, control, lineage)
    if len(a) != 57:
        raise ValueError('Unexpected selected size')
    output.mkdir(exist_ok=False)
    files = {}
    for name, rows in [('context.jsonl', a), ('last-user.jsonl', b), ('lineage.jsonl', meta)]:
        data = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n' for row in rows).encode('utf8')
        (output / name).write_bytes(data)
        files[name] = sha(data)
    result = {'format': 'matched-dialogue-v1', 'status': 'reviewed_candidate_not_approved_for_training',
              'limitations': ['Model review is not human, factual, privacy, or rights certification.',
                              'One chosen pair per reviewed group; not representative of all records.',
                              'Insufficient size for broad chat training or evidence of generalization.'],
              'tokenizer_sha256': manifest['tokenizer_sha256'], 'input_sha256': PINS,
              'builder_sha256': sha(Path(__file__).read_bytes()), 'files': files,
              'counts': {'pairs': len(a), 'groups': len({r['group'] for r in a}),
                         'target_tokens': sum(sum(r['mask']) for r in a)}}
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=RUNS / 'p108-reviewed-pairs-v1')
    build(parser.parse_args().output)
