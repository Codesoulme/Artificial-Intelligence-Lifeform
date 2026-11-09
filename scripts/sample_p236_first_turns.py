"""Sample first turns from long dialogues without reusing reviewed families."""
import hashlib
import json
from pathlib import Path

from scripts import prepare_p231_dialogue_candidates as previous
from scripts.sample_p235_long_dialogues import INPUT, rank
from tokenizers import Tokenizer

ROOT = previous.ROOT
OUTPUT = ROOT/'runs/p236-first-turn-review-20261109'
PINS = {
    'candidate': 'f3039fbb68fe0af14fc135e3c42dbeaf5fc4c604f0c6997fe805571bdaae21a3',
    'p231': '5998f39f850c74673cb0e5ae59bc391765803aac39d49e4c7542ddc242619cd4',
    'p235': '1a249afc18b644d94612904844f7dfea8465b235aa28c73e1bc8be6e07fc59d2',
}


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = {'candidate': INPUT/'candidates.jsonl',
             'p231': ROOT/'runs/p231-dialogue-candidates/review-40.jsonl',
             'p235': ROOT/'runs/p235-long-dialogue-review-20261109/sample.jsonl',
             'source': ROOT/'runs/conversation-data-v5/dialogue-train.jsonl',
             'tokenizer': ROOT/'runs/conversation-data-v5/tokenizer.json'}
    expected = {**PINS, 'source': previous.EXPECTED['source'], 'tokenizer': previous.EXPECTED['tokenizer']}
    if any(previous.file_hash(paths[key]) != value for key, value in expected.items()):
        raise ValueError('Input digest mismatch')
    excluded = {json.loads(line)['family_sha256'] for key in ('p231', 'p235')
                for line in paths[key].read_bytes().splitlines()}
    by_family = {}
    for line in paths['candidate'].read_bytes().splitlines():
        row = json.loads(line)
        if row['answer_message_index'] == 1 and row['family_sha256'] not in excluded:
            by_family.setdefault(row['family_sha256'], []).append(row)
    chosen = [min(by_family[f], key=lambda r: rank('p236-dialogue', r['dialogue_sha256']))
              for f in sorted(by_family, key=lambda f: rank('p236-family', f))[:20]]
    if len(chosen) != 20:
        raise ValueError('Insufficient disjoint families')
    wanted = {r['source_line'] for r in chosen}
    sources = {}
    hasher = hashlib.sha256()
    with paths['source'].open('rb') as stream:
        for n, line in enumerate(stream, 1):
            hasher.update(line)
            if n in wanted:
                sources[n] = json.loads(line)
    if hasher.hexdigest() != expected['source'] or len(sources) != len(wanted):
        raise ValueError('Source changed or rows missing')
    tokenizer = Tokenizer.from_file(str(paths['tokenizer']))
    sample = []
    for index, row in enumerate(chosen):
        messages = sources[row['source_line']]['messages']
        if previous.dialogue_hash(messages) != row['dialogue_sha256']:
            raise ValueError('Source dialogue mismatch')
        target = tokenizer.decode([t for t, enabled in zip(row['tokens'], row['mask']) if enabled], skip_special_tokens=False)
        if target != messages[1]['content']+'<|eos|>':
            raise ValueError('Target differs from complete answer')
        sample.append({'index': index, 'family_sha256': row['family_sha256'], 'dialogue_sha256': row['dialogue_sha256'],
                       'messages': messages[:2], 'target_tokens': row['target_tokens']})
    payload = b''.join((json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8') for row in sample)
    OUTPUT.mkdir(parents=False, exist_ok=False)
    (OUTPUT/'sample.jsonl').write_bytes(payload)
    manifest = {'status': 'review_only_not_training', 'sampling': 'one first turn per family; P231 and P235 review families excluded',
                'ranking': 'SHA256 with p235-review-20261109-v1 seed and p236-family / p236-dialogue domains',
                'eligible_families': len(by_family), 'samples': len(sample), 'input_sha256': expected,
                'script_sha256': previous.file_hash(Path(__file__)), 'sample_sha256': hashlib.sha256(payload).hexdigest(),
                'limitations': ['First turns may themselves lack context or be inaccurate.',
                                'Sample is family-weighted and model review is not human or fact checking.',
                                'Prior reviews other than P231 and P235 may have covered a family.']}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False, indent=2))
