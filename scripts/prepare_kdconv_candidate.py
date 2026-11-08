"""Prepare KdConv train-only dialogue candidates, never approve them automatically."""
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import encode_dialogue, normalized, render
from scripts.prepare_p28_prose import HeldIndex, file_hash, load_old_held
from scripts.fetch_kdconv_train import FILES, REPOSITORY, REVISION

MATERIAL = ROOT/'runs/kdconv-material'
OUTPUT = ROOT/'runs/kdconv-dialogue-candidate-v1'
REPORT = ROOT/'reports/P35-kdconv-preparation.json'
TOKENIZER = ROOT/'runs/conversation-data-v5/tokenizer.json'
SPECIAL = re.compile(r'<\||\ufffd|https?://|```|[\x00-\x08\x0b\x0c\x0e-\x1f]')
IDENTIFIERS = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
PERSONAL = re.compile(r'我(?:曾|去过|吃过|看过|住过|买过|遇到过|亲身|最近|昨天|今天|家人|朋友|小时候|上次)')
REPEAT = re.compile(r'(.{2,20})\1{3,}')


def main():
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError('Candidate or report already exists')
    provenance = json.loads((MATERIAL/'provenance.json').read_text('utf-8'))
    if provenance.get('split') != 'train' or provenance.get('ready_for_training') is not False:
        raise ValueError('Expected unapproved train-only provenance')
    if (provenance.get('repository') != REPOSITORY or provenance.get('revision') != REVISION
            or [item.get('path') for item in provenance.get('files', [])] != list(FILES)):
        raise ValueError('Unexpected source repository, revision, or file list')
    data_manifest = json.loads((TOKENIZER.parent/'manifest.json').read_text('utf-8'))
    if file_hash(TOKENIZER) != data_manifest['files']['tokenizer.json']:
        raise ValueError('Tokenizer changed')
    for item in provenance['files']:
        path = MATERIAL/item['path']
        if file_hash(path) != item['sha256']:
            raise ValueError('Source file changed: '+item['path'])
    old_held, evidence = load_old_held()
    dev_cases = json.loads((ROOT/'data/conversation-development.json').read_text('utf-8'))
    dev_questions = {normalized(t) for scenario in dev_cases['scenarios'] for t in scenario['turns']}
    for question in dev_questions:
        old_held.add(question)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    stats = Counter()
    seen = set()
    OUTPUT.mkdir(parents=True)
    with (OUTPUT/'dialogue.jsonl').open('x', encoding='utf-8') as raw, \
         (OUTPUT/'dialogue-tokens.jsonl').open('x', encoding='utf-8') as encoded:
        for item in provenance['files']:
            records = json.loads((MATERIAL/item['path']).read_text('utf-8'))
            for index, record in enumerate(records):
                stats['source_dialogues'] += 1
                turns = record.get('messages')
                if not isinstance(turns, list):
                    stats['bad_record'] += 1
                    continue
                group = hashlib.sha256((item['path']+':'+str(index)).encode()).hexdigest()
                # Fixed nonoverlapping three-pair windows preserve source order.
                for start in range(0, len(turns)-3, 6):
                    window = turns[start:start+6]
                    if len(window) % 2:
                        window = window[:-1]
                    if len(window) < 4:
                        continue
                    stats['windows_considered'] += 1
                    texts = [t.get('message') if isinstance(t, dict) else None for t in window]
                    if any(not isinstance(t, str) or not t.strip() or SPECIAL.search(t) or IDENTIFIERS.search(t) for t in texts):
                        stats['malformed_or_identifier'] += 1
                        continue
                    messages = [dict(role='user' if n%2 == 0 else 'assistant', content=t.strip()) for n,t in enumerate(texts)]
                    if any(PERSONAL.search(m['content']) for m in messages if m['role'] == 'assistant'):
                        stats['possible_assistant_experience'] += 1
                        continue
                    if any(REPEAT.search(m['content']) for m in messages):
                        stats['repetition'] += 1
                        continue
                    if any(normalized(m['content']) in dev_questions or old_held.reason(m['content']) for m in messages):
                        stats['existing_held_overlap'] += 1
                        continue
                    text = render(messages)
                    if old_held.reason(text):
                        stats['existing_held_overlap'] += 1
                        continue
                    key = hashlib.sha256(normalized(text).encode('utf-8')).hexdigest()
                    if key in seen:
                        stats['duplicate'] += 1
                        continue
                    result = encode_dialogue(tokenizer, messages, limit=512)
                    if result is None:
                        stats['overlength'] += 1
                        continue
                    seen.add(key)
                    tokens, mask = result
                    row = dict(messages=messages, text=text, group=group, sha256=key,
                               source_repository=provenance['repository'], source_revision=provenance['revision'],
                               source_file=item['path'], source_dialogue_index=index, source_turn_offset=start,
                               role_assignment='Alternating source speakers; even window turn is user, odd is assistant. Both are human-authored.')
                    raw.write(json.dumps(row, ensure_ascii=False)+'\n')
                    encoded.write(json.dumps(dict(tokens=tokens, mask=mask, group=group))+'\n')
                    stats['windows_kept'] += 1
                    stats['assistant_target_tokens'] += sum(mask)
    result = dict(status='prepared_pending_review', ready_for_training=False,
                  source_repository=provenance['repository'], source_revision=provenance['revision'],
                  source_provenance_sha256=file_hash(MATERIAL/'provenance.json'),
                  tokenizer_sha256=file_hash(TOKENIZER), stats=dict(stats),
                  old_held_evidence=evidence, development_prompts_sha256=file_hash(ROOT/'data/conversation-development.json'),
                  files={name:file_hash(OUTPUT/name) for name in ('dialogue.jsonl','dialogue-tokens.jsonl')},
                  limits=['Existing held checks cover exact/group and finite sampled shingles, not semantic equivalence.',
                          'Both speakers are humans; assigning one assistant role may teach personal claims. Regex is insufficient to remove them.',
                          'Source knowledge attributes are omitted; dialogue may depend on missing information.',
                          'Knowledge facts are not independently verified; current-source QA and future generalization are separate.',
                          'Windows within each original dialogue share a group; never split the group across train/development.',
                          'No original KdConv development or test data was read.'])
    (OUTPUT/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['stats']), flush=True)


if __name__ == '__main__':
    main()
