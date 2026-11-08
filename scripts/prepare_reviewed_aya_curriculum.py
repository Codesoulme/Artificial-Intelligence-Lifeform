"""Construct a quarantined reviewed-QA plus existing-dialogue curriculum."""
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import encode_dialogue, sha
from scripts.prepare_aya_candidate import normalized
from spora.qa_cleanup import strip_qa_labels

REVIEW = ROOT/'runs/aya-review-p83'
AYA = ROOT/'runs/aya-chinese-candidate-v1'
BASE = ROOT/'runs/mixed-context-curriculum-v1'
VOCAB = ROOT/'runs/p28-stories-data/candidate-v3/tokenizer.json'
OLD = ROOT/'runs/conversation-data-v5/dialogue-train.jsonl'
OUT = ROOT/'runs/reviewed-aya-curriculum-p83-v2'
DEVS = [ROOT/'data/conversation-development.json', ROOT/'data/conversation-development-v2.json']


def rows(path):
    with path.open(encoding='utf8') as f:
        return [json.loads(line) for line in f]


def near(a, b):
    if len(a) < 8 or len(b) < 8:
        return False
    if a in b or b in a:
        return True
    x = {a[i:i+5] for i in range(len(a)-4)}
    y = {b[i:i+5] for i in range(len(b)-4)}
    return bool(x and y and len(x & y)/len(x | y) >= .5)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    selection = json.loads((REVIEW/'manifest.json').read_text(encoding='utf8'))
    aya_manifest = json.loads((AYA/'manifest.json').read_text(encoding='utf8'))
    if sha(AYA/'train.jsonl') != selection['source_sha256'] or sha(AYA/'manifest.json') != selection['candidate_manifest_sha256']:
        raise ValueError('Reviewed candidate changed')
    base_manifest = json.loads((BASE/'manifest.json').read_text(encoding='utf8'))
    for name,digest in base_manifest['files'].items():
        if sha(BASE/name) != digest:
            raise ValueError('Original curriculum changed: '+name)
    old_manifest = json.loads((OLD.parent/'manifest.json').read_text(encoding='utf8'))
    if sha(OLD) != old_manifest['files'][OLD.name]:
        raise ValueError('Prior dialogue source changed')
    reviews, record_by_index, accepted = {}, {}, set()
    for n in range(1,5):
        path = REVIEW/f'shard-{n}.jsonl'
        if sha(path) != selection['files'][path.name]:
            raise ValueError('Review shard changed')
        records = {x['source_index']:x for x in rows(path)}
        judged = rows(REVIEW/f'review-{n}.jsonl')
        if len(records) != 60 or len(judged) != 60 or len({x['source_index'] for x in judged}) != 60:
            raise ValueError('Incomplete first review')
        for verdict in judged:
            index = verdict['source_index']
            if index in record_by_index or index not in records:
                raise ValueError('Unexpected or duplicate source row')
            record = records[index]
            canonical = json.dumps(record['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf8')
            if hashlib.sha256(canonical).hexdigest() != record['messages_sha256'] or verdict['messages_sha256'] != record['messages_sha256']:
                raise ValueError('Review does not bind the message text')
            if verdict['decision'] not in ('accept','reject') or len(verdict.get('reason','').strip()) < 2:
                raise ValueError('Invalid first review')
            record_by_index[index] = record
            if verdict['decision'] == 'accept':
                accepted.add(index)
        reviews[path.name] = sha(path)
        reviews[f'review-{n}.jsonl'] = sha(REVIEW/f'review-{n}.jsonl')
    cross_seen, agreed = set(), set()
    for n in (1,2):
        path, verdict_path = REVIEW/f'cross-shard-{n}.jsonl', REVIEW/f'cross-review-{n}.jsonl'
        expected = {x['source_index']:x for x in rows(path)}
        judged = rows(verdict_path)
        if len(expected) != len(judged) or len({x['source_index'] for x in judged}) != len(expected):
            raise ValueError('Incomplete cross review')
        for verdict in judged:
            i = verdict['source_index']
            if i in cross_seen or i not in expected or i not in accepted:
                raise ValueError('Cross review covers wrong row')
            if expected[i] != record_by_index[i] or verdict['messages_sha256'] != expected[i]['messages_sha256']:
                raise ValueError('Cross review text differs')
            if verdict['decision'] not in ('accept','reject') or len(verdict.get('reason','').strip()) < 2:
                raise ValueError('Invalid cross review')
            cross_seen.add(i)
            if verdict['decision'] == 'accept':
                agreed.add(i)
        reviews[path.name], reviews[verdict_path.name] = sha(path), sha(verdict_path)
    if cross_seen != accepted:
        raise ValueError('Cross review did not cover every accepted row')
    old_questions = set()
    with OLD.open(encoding='utf8') as f:
        for line in f:
            for message in json.loads(line).get('messages',[]):
                if message.get('role') == 'user':
                    old_questions.add(normalized(message.get('content','')))
    dev_prompts = []
    for path in DEVS:
        data = json.loads(path.read_text(encoding='utf8'))
        dev_prompts.extend(normalized(q) for scenario in data['scenarios'] for q in scenario['turns'])
    tokenizer = Tokenizer.from_file(str(VOCAB))
    chosen, holdout, counts = [], [], collections.Counter()
    for index in sorted(agreed):
        item = record_by_index[index]
        original = item['messages']
        q,a = strip_qa_labels(original[0]['content'],original[1]['content'])
        normalized_q = normalized(q)
        if normalized_q in old_questions:
            counts['exact_prior_dialogue_prompt'] += 1
            continue
        if any(near(normalized_q, prompt) for prompt in dev_prompts):
            counts['possible_development_overlap'] += 1
            continue
        messages = [{'role':'user','content':q},{'role':'assistant','content':a}]
        value = encode_dialogue(tokenizer,messages,limit=512)
        if value is None:
            counts['token_length'] += 1
            continue
        ids,mask = value
        raw = dict(group='aya-'+item['group'],messages=messages,source='Aya Dataset, Apache-2.0',
                   source_index=index,original_messages_sha256=item['messages_sha256'])
        encoded = dict(group=raw['group'],tokens=ids,mask=mask)
        counts['labels_removed_question'] += int(q != original[0]['content'].strip())
        counts['labels_removed_answer'] += int(a != original[1]['content'].strip())
        split_key = hashlib.sha256(('p83-holdout:'+item['group']).encode('ascii')).hexdigest()
        target = holdout if int(split_key[:8],16)%10 == 0 else chosen
        target.append((raw,encoded))
    holdout_questions = {normalized(raw['messages'][0]['content']) for raw,_ in holdout}
    unique_questions, deduplicated = set(), []
    for raw, encoded in chosen:
        q = normalized(raw['messages'][0]['content'])
        if q in holdout_questions:
            counts['cleaned_prompt_matches_holdout'] += 1
        elif q in unique_questions:
            counts['duplicate_cleaned_training_prompt'] += 1
        else:
            unique_questions.add(q)
            deduplicated.append((raw,encoded))
    chosen = deduplicated
    if len(chosen)<40 or not holdout:
        raise ValueError(f'Too few reviewed QA records: train={len(chosen)}, development={len(holdout)}, exclusions={dict(counts)}')
    original_raw, original_token = rows(BASE/'teacher.jsonl'),rows(BASE/'teacher-tokens.jsonl')
    if len(original_raw)!=24 or len(original_token)!=24 or any(a['group']!=b['group'] for a,b in zip(original_raw,original_token)):
        raise ValueError('Original reviewed multi-turn data differs')
    OUT.mkdir(parents=True)
    for name,items in [('teacher',list(zip(original_raw,original_token))+chosen),('aya-development',holdout)]:
        for suffix,column in (('.jsonl',0),('-tokens.jsonl',1)):
            with (OUT/(name+suffix)).open('x',encoding='utf8') as f:
                for item in items:f.write(json.dumps(item[column],ensure_ascii=False)+'\n')
    for name in ('foundations.jsonl','foundations-tokens.jsonl'):
        shutil.copy2(BASE/name,OUT/name)
    manifest = dict(status='candidate_pending_final_review_not_approved_for_training',
                    tokenizer_sha256=sha(VOCAB),source_repository='CohereForAI/aya_dataset',
                    source_revision='f9ea04583f02a8f86404ff6c58bf75fe637df8a2',declared_license='Apache-2.0',
                    generator_sha256=sha(Path(__file__)),label_cleanup_sha256=sha(ROOT/'spora/qa_cleanup.py'),
                    initial_review_accepts=len(accepted),cross_review_accepts=len(agreed),
                    review_artifact_sha256=reviews,source_candidate_manifest_sha256=sha(AYA/'manifest.json'),
                    old_curriculum_manifest_sha256=sha(BASE/'manifest.json'),old_dialogue_sha256=sha(OLD),
                    development_case_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in DEVS},
                    counts=dict(counts),authored_multiturn=len(original_raw),aya_train=len(chosen),aya_development=len(holdout),
                    aya_train_targets=sum(sum(x[1]['mask']) for x in chosen),
                    files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
                    limitations=['All Aya records are single-turn and reviewed by two AI instances, not human experts.',
                                 'Review does not guarantee factual accuracy; cautious accepted examples may remain imperfect.',
                                 'Explicit leading QA labels removed; all other message text preserved. Cleaned duplicate train prompts removed.',
                                 'Cross-source near-duplicates and final-test isolation are not fully resolved.',
                                 'Development sets are for model selection, not final chat certification.'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('review_artifact_sha256','files')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
