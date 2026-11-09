"""Prepare a blinded queue from two existing development conversation runs."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'p223': ('runs/p230-p223-multiturn-v2-20261109.json',
             'e5df3ba8c17c8fd2c5e0d50d687cab69a72136bb9fe0ca5bf7bda973298d9cd8', 10),
    'p229': ('runs/p230-p229-multiturn-v2-20261109.json',
             'ba4cb45323ff3ca48bb825439d3eaa3af64bf8ba078bcbe4c8649eb6b18ffecc', 100),
}
CASES_SHA256 = 'fac3f3089c94b3ecdc29ffa98c1d24b938ab5d13717c275b53c7f2d38ca4d9c5'
SHARED_FIELDS = ('plan_sha256', 'policy', 'sampling', 'tokenizer_sha256',
                 'evaluator_sha256', 'generation_source_sha256', 'repetition_metric_version')
SEED = 23020261109


def load_source(root, name):
    path, checkpoint, step = SOURCES[name]
    raw = (root / path).read_bytes()
    data = json.loads(raw)
    if (data.get('checkpoint_sha256') != checkpoint or data.get('step') != step
            or data.get('cases_sha256') != CASES_SHA256):
        raise ValueError(f'{name}: unexpected checkpoint, step or cases digest')
    if (data.get('checkpoint_unchanged') is not True or data.get('old_individual_exact') is not True
            or data.get('summary', {}).get('status') != 'ungraded_development_not_acceptance'
            or data['summary'].get('turns') != 24):
        raise ValueError(f'{name}: invalid development run status')
    rows = data.get('rows')
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError(f'{name}: expected 24 rows')
    conversations = {}
    for offset in range(0, 24, 3):
        group = rows[offset:offset + 3]
        if not all(isinstance(row, dict) for row in group):
            raise ValueError(f'{name}: invalid row')
        scenario = group[0].get('scenario')
        if (not isinstance(scenario, str) or not scenario or scenario in conversations
                or any(row.get('scenario') != scenario or type(row.get('turn')) is not int
                       or row['turn'] != index for index, row in enumerate(group, 1))
                or any(not isinstance(row.get(key), str) or not row[key].strip()
                       for row in group for key in ('prompt', 'text'))):
            raise ValueError(f'{name}: invalid scenario or turns')
        conversations[scenario] = [{'prompt': row['prompt'], 'text': row['text']} for row in group]
    return data, conversations, hashlib.sha256(raw).hexdigest()


def build(root, output_dir):
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.parent != (root / 'runs').resolve():
        raise ValueError('output directory must be a new direct child of runs')
    loaded = {name: load_source(root, name) for name in SOURCES}
    first, second = (loaded[name] for name in SOURCES)
    if any(field not in first[0] or field not in second[0] or first[0][field] != second[0][field]
           for field in SHARED_FIELDS):
        raise ValueError('generation settings differ')
    if first[1].keys() != second[1].keys():
        raise ValueError('scenarios differ')
    for scenario in first[1]:
        if [row['prompt'] for row in first[1][scenario]] != [row['prompt'] for row in second[1][scenario]]:
            raise ValueError(f'prompts differ for {scenario}')

    rng = random.Random(SEED)
    entries = [(name, scenario, turns) for name, (_, cases, _) in loaded.items()
               for scenario, turns in cases.items()]
    rng.shuffle(entries)
    ids = rng.sample(range(10**8, 10**9), len(entries))
    queue, mapping = [], {}
    for number, (name, scenario, turns) in zip(ids, entries):
        anonymous = f'{number:09d}'
        queue.append({'anonymous_id': anonymous, 'turns': turns})
        mapping[anonymous] = {'source': name, 'scenario': scenario}
    manifest = {
        'scope': 'development_only_not_independent_final_test',
        'seed': SEED,
        'inputs': {name: {'path': SOURCES[name][0], 'sha256': loaded[name][2],
                          'checkpoint_sha256': SOURCES[name][1], 'cases_sha256': CASES_SHA256}
                   for name in SOURCES},
        'mapping': mapping,
    }
    output_dir.mkdir(exist_ok=False)
    for filename, value in (('blind.json', queue), ('local_mapping.json', manifest)):
        with (output_dir / filename).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='new direct child directory under runs')
    args = parser.parse_args()
    print(build(ROOT, ROOT / args.output_dir))


if __name__ == '__main__':
    main()
