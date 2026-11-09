"""Create a local, family-disjoint review sample from P231 candidates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/p237-dialogue-review-20261109'
PATHS = {
    'manifest': ROOT / 'runs/p231-dialogue-candidates/manifest.json',
    'candidate': ROOT / 'runs/p231-dialogue-candidates/candidates.jsonl',
    'p231': ROOT / 'runs/p231-dialogue-candidates/review-40.jsonl',
    'p235': ROOT / 'runs/p235-long-dialogue-review-20261109/sample.jsonl',
    'p236': ROOT / 'runs/p236-first-turn-review-20261109/sample.jsonl',
}
PINS = {
    'manifest': '160fa3560a7b6a113a240777cc6cb9142987e098aaecbb2d65ac0fa610a34a61',
    'candidate': '3f7ba78716d438d0e05cd40ccf90be6e0fbabea934ce641d0d1589c5bbe47ab1',
    'p231': '5998f39f850c74673cb0e5ae59bc391765803aac39d49e4c7542ddc242619cd4',
    'p235': '1a249afc18b644d94612904844f7dfea8465b235aa28c73e1bc8be6e07fc59d2',
    'p236': '78f5ddcd553e9c6a2c810199bf21f0cff2794516107c1e24d20e8c46626de29c',
}
SEED = 'p237-dialogue-review-20261109-v1'
STATUS = 'pending_quality_review_not_approved_for_training'
COUNT = 40


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rank(domain, value):
    return digest((SEED + '\0' + domain + '\0' + value).encode('utf-8'))


def validate_row(row):
    if row.get('status') != STATUS:
        raise ValueError('Invalid candidate status')
    tokens, mask = row['tokens'], row['mask']
    if (not isinstance(tokens, list) or not isinstance(mask, list)
            or not 0 < len(tokens) <= 512 or len(tokens) != len(mask)
            or any(type(t) is not int or t < 0 for t in tokens)
            or any(type(m) not in (bool, int) or m not in (0, 1) for m in mask)
            or row['token_count'] != len(tokens)
            or row['supervised_tokens'] != sum(mask) or not sum(mask)):
        raise ValueError('Invalid candidate encoding')
    messages = row['messages']
    if (not isinstance(messages, list) or len(messages) < 4
            or any(not isinstance(m, dict) or m.get('role') not in ('user', 'assistant')
                   or not isinstance(m.get('content'), str) for m in messages)):
        raise ValueError('Invalid candidate messages')
    for field in ('family_sha256', 'dialogue_sha256'):
        value = row[field]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid candidate digest')


def select(rows, excluded, count=COUNT):
    families = {}
    seen = set()
    for row in rows:
        validate_row(row)
        key = (row['family_sha256'], row['dialogue_sha256'])
        if key in seen:
            raise ValueError('Duplicate candidate dialogue')
        seen.add(key)
        if key[0] not in excluded:
            families.setdefault(key[0], []).append(row)
    if len(families) < count:
        raise ValueError('Insufficient disjoint families')
    chosen = [min(families[family], key=lambda row: (rank('dialogue', row['dialogue_sha256']), row['dialogue_sha256']))
              for family in sorted(families, key=lambda family: (rank('family', family), family))[:count]]
    return chosen, len(families)


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = {}
    for name, path in PATHS.items():
        data[name] = path.read_bytes()
        if digest(data[name]) != PINS[name]:
            raise ValueError(name + ' digest mismatch')
    manifest = json.loads(data['manifest'])
    if (manifest['status'] != STATUS
            or manifest['output_sha256']['candidates.jsonl'] != PINS['candidate']
            or manifest['output_sha256']['review-40.jsonl'] != PINS['p231']):
        raise ValueError('Candidate manifest mismatch')
    prior = {name: [json.loads(line) for line in data[name].splitlines()] for name in ('p231', 'p235', 'p236')}
    for name, expected in (('p231', 40), ('p235', 20), ('p236', 20)):
        families = [row['family_sha256'] for row in prior[name]]
        if len(families) != expected or len(set(families)) != expected:
            raise ValueError(name + ' sample count or families invalid')
    excluded = {row['family_sha256'] for rows in prior.values() for row in rows}
    rows = [json.loads(line) for line in data['candidate'].splitlines()]
    if (len(rows) != manifest['counts']['retained_dialogues']
            or len({row['family_sha256'] for row in rows}) != manifest['counts']['retained_families']):
        raise ValueError('Candidate count mismatch')
    chosen, eligible = select(rows, excluded)
    payload = b''.join((json.dumps({'index': i, **row}, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
                       for i, row in enumerate(chosen))
    result = {
        'status': 'review_only_not_training', 'seed': SEED, 'samples': len(chosen),
        'eligible_families': eligible, 'excluded_families': len(excluded),
        'sampling': 'SHA256-ranked families, then one SHA256-ranked complete dialogue per family',
        'input_sha256': PINS, 'script_sha256': digest(Path(__file__).read_bytes()),
        'sample_sha256': digest(payload),
        'limitations': ['For local quality review only; not approved for training.',
                        'Family-weighted selection is not uniform over dialogues.',
                        'Only P231, P235 and P236 review families are excluded; other overlap may remain.',
                        'Family isolation and heuristic screening do not guarantee privacy, safety, accuracy or rights.',
                        'Full dialogue content is sensitive and must remain local.'],
    }
    OUTPUT.mkdir(parents=False, exist_ok=False)
    with (OUTPUT / 'sample.jsonl').open('xb') as stream:
        stream.write(payload)
    with (OUTPUT / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
