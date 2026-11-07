"""Prepare complete, short multi-turn TRAIN examples from the full source.

This supplemental material is not selected by any current training plan.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import ROOT, sha, conversation, render, normalized, split_of, encode_dialogue
from scripts.prepare_conversation_curriculum import exclusion
from scripts.curate_conversation import reject
from scripts.audit_conversation_data import signatures


def main():
    source = ROOT / 'runs/language-material/sft_t2t_mini.jsonl'
    provenance = json.loads((source.parent / 'dialogue-provenance.json').read_text('utf-8'))
    if sha(source) != provenance['sha256']:
        raise ValueError('Source changed')
    data = ROOT / 'runs/conversation-data-v4'
    tokenizer = Tokenizer.from_file(str(data / 'tokenizer.json'))
    held_sizes = []; index = collections.defaultdict(list)
    groups, digests, prompts = set(), set(), set()
    for directory in ('conversation-data-v3', 'conversation-data-v4'):
        for split in ('development', 'test'):
            for kind in ('pretrain', 'dialogue'):
                with (ROOT / 'runs' / directory / f'{kind}-{split}.jsonl').open(encoding='utf-8') as stream:
                    for line in stream:
                        row = json.loads(line); groups.add(row['group']); digests.add(row['sha256'])
                        sigs = signatures(row['text']); number = len(held_sizes); held_sizes.append(len(sigs))
                        for sig in sigs: index[sig].append(number)
                        prompts.update(normalized(m['content']) for m in row.get('messages', ()) if m['role'] == 'user')
    development = json.loads((ROOT / 'data/conversation-development.json').read_text('utf-8'))
    prompts.update(normalized(t) for c in development['scenarios'] for t in c['turns'])
    from scripts.train_conversation_v6 import PROMPTS
    prompts.update(map(normalized, PROMPTS))
    out = ROOT / 'runs/multiturn-curriculum-v1'; out.mkdir(exist_ok=False)
    stats = collections.Counter(); seen = set()
    with source.open('rb') as stream, (out / 'multiturn.jsonl').open('x', encoding='utf-8') as raw, (out / 'multiturn-tokens.jsonl').open('x', encoding='utf-8') as encoded:
        for number, line in enumerate(stream):
            stats['scanned'] += 1
            messages = conversation(json.loads(line))
            if messages is None or not 4 <= len(messages) <= 10:
                stats['not_complete_2_to_5_turns'] += 1; continue
            if split_of(messages[0]['content']) != 'train':
                stats['non_training_partition'] += 1; continue
            text = render(messages)
            row = dict(messages=messages, text=text, source_line=number,
                       sha256=hashlib.sha256(normalized(text).encode()).hexdigest(),
                       group=hashlib.sha256(normalized(messages[0]['content'])[:96].encode()).hexdigest())
            why = reject(row, 'dialogue') or exclusion(messages)
            if why:
                stats['removed_' + why] += 1; continue
            answers = [m['content'] for m in messages if m['role'] == 'assistant']
            if any(len(s) > 200 or '```' in s or '\ufffd' in s for s in answers):
                stats['long_or_code_answer'] += 1; continue
            if any(len(s) > 30 and not re.search(r'[。！？!?；;」』”）)\]】.]$', s.rstrip()) for s in answers):
                stats['unfinished_answer'] += 1; continue
            if re.search(r'诊断|治疗|药物|剂量|处方|诉讼|股票|股价|投资|汇率|疾病|病人', text):
                stats['advanced_topic'] += 1; continue
            if row['sha256'] in seen:
                stats['duplicate'] += 1; continue
            if row['group'] in groups or row['sha256'] in digests or any(normalized(m['content']) in prompts for m in messages if m['role'] == 'user'):
                stats['held_exact_group_or_prompt'] += 1; continue
            value = encode_dialogue(tokenizer, messages, limit=512)
            if value is None:
                stats['overlength'] += 1; continue
            sigs = signatures(text); shared = collections.Counter(i for s in sigs for i in index.get(s, ()))
            if any(min(len(sigs), held_sizes[i]) >= 8 and n / min(len(sigs), held_sizes[i]) >= .8 for i, n in shared.items()):
                stats['held_lexical_overlap'] += 1; continue
            seen.add(row['sha256']); tokens, mask = value
            raw.write(json.dumps(row, ensure_ascii=False) + '\n')
            encoded.write(json.dumps(dict(tokens=tokens, mask=mask, group=row['group'])) + '\n')
            stats['dialogues'] += 1; stats['turns'] += len(messages) // 2; stats['targets'] += sum(mask)
            stats[f'{len(messages)//2}_turn_dialogues'] += 1
            if stats['dialogues'] % 5000 == 0:
                print(json.dumps(dict(stage='selected', stats=stats)), flush=True)
    manifest = dict(stats=stats, source_provenance=provenance, tokenizer_sha256=sha(data / 'tokenizer.json'),
                    source_code_sha256={s: sha(ROOT/s) for s in ('scripts/prepare_multiturn_curriculum.py', 'scripts/prepare_conversation_curriculum.py', 'scripts/curate_conversation.py', 'scripts/prepare_conversation_v2.py', 'scripts/audit_conversation_data.py')},
                    held_manifest_sha256={d: sha(ROOT/'runs'/d/'manifest.json') for d in ('conversation-data-v3', 'conversation-data-v4')},
                    scope='New TRAIN-only short multi-turn subset; no truncation, no new tokenizer. No factual or semantic-isolation guarantee; requires raw review before use. Not individual experiences.',
                    files={p.name: sha(p) for p in out.iterdir()})
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(stage='completed', stats=stats)), flush=True)


if __name__ == '__main__':
    main()
