"""Training-only short-dialogue curriculum and explicit authored teaching."""
import collections,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import ROOT,sha,normalized,encode_dialogue,render

ABSTRACT=re.compile(r'人工智能|量子|自我意识|自我存在|跨文化|全球治理|算法伦理|文化语境|\bAI\b',re.I)
FIRST_PERSON=re.compile(r'(我|我们).{0,12}(今天|昨天|最近|童年|小时候|朋友|家人|同事|吃过|吃了|喝了|去了|读过|看过|住在|出生|大学|毕业)')

def exclusion(messages):
    if ABSTRACT.search(messages[0]['content']):return 'abstract_topic'
    for message in messages:
        if message['role']!='assistant':continue
        s=message['content'];han=sum('\u4e00'<=c<='\u9fff' for c in s);latin=sum('a'<=c.lower()<='z' for c in s)
        if han<.92*(han+latin):return 'latin_heavy'
        if len(s)>300:return 'long_answer'
        if FIRST_PERSON.search(s):return 'possible_personal_experience'
        if re.search(r'(.{2,20})\1{2,}',s):return 'consecutive_repetition'
        grams=[s[i:i+6] for i in range(max(0,len(s)-5))]
        if len(grams)>40 and 1-len(set(grams))/len(grams)>.2:return 'repeated_phrases'
        if s.count('**')>8 or s.count('#')>3:return 'heavy_formatting'
    return None

def main():
    source=ROOT/'runs/conversation-data-v4';out=ROOT/'runs/conversation-curriculum-v1';out.mkdir(exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text('utf-8'));stats=collections.Counter()
    for name in ('dialogue-train.jsonl','tokenizer.json'):
        if sha(source/name)!=manifest['files'][name]:raise ValueError('Source changed')
    tok=Tokenizer.from_file(str(source/'tokenizer.json'));records=[];seen=set();groups=set()
    with (source/'dialogue-train.jsonl').open(encoding='utf-8') as f,(out/'short-train-tokens.jsonl').open('x',encoding='utf-8') as encoded,(out/'short-train.jsonl').open('x',encoding='utf-8') as raw:
        for line in f:
            row=json.loads(line);why=exclusion(row['messages'])
            if why:stats['excluded_'+why]+=1;continue
            value=encode_dialogue(tok,row['messages'],limit=384)
            if value is None:stats['excluded_length']+=1;continue
            tokens,mask=value;encoded.write(json.dumps(dict(tokens=tokens,mask=mask,group=row['group']))+'\n');raw.write(line)
            stats['short_dialogues']+=1;stats['short_targets']+=sum(mask);stats['short_turns']+=len(row['messages'])//2
            seen.add(row['sha256']);groups.add(row['group'])
    authored=ROOT/'data/chinese-foundations.json';teach=json.loads(authored.read_text('utf-8'))
    with (out/'foundations-tokens.jsonl').open('x',encoding='utf-8') as encoded,(out/'foundations.jsonl').open('x',encoding='utf-8') as raw:
        for number,dialogue in enumerate(teach['dialogues']):
            messages=[dict(role=role,content=text) for pair in dialogue['turns'] for role,text in zip(('user','assistant'),pair)]
            value=encode_dialogue(tok,messages,limit=512)
            if value is None:raise ValueError('Authored dialogue too long; rewrite instead of truncate')
            text=render(messages);digest=hashlib.sha256(normalized(text).encode()).hexdigest()
            if digest in seen:raise ValueError('Duplicate authored dialog')
            seen.add(digest);group=hashlib.sha256(normalized(messages[0]['content'])[:96].encode()).hexdigest()
            tokens,mask=value;encoded.write(json.dumps(dict(tokens=tokens,mask=mask,group=group))+'\n')
            raw.write(json.dumps(dict(topic=dialogue['topic'],messages=messages,text=text,sha256=digest,group=group),ensure_ascii=False)+'\n')
            stats['foundation_dialogues']+=1;stats['foundation_turns']+=len(dialogue['turns']);stats['foundation_targets']+=sum(mask)
    result=dict(source_manifest_sha256=sha(source/'manifest.json'),tokenizer_sha256=sha(source/'tokenizer.json'),
        authored_sha256=sha(authored),source_sha256=sha(Path(__file__)),stats=stats,
        scope='TRAIN subset only; held data unchanged. Added authored teaching is never final assessment. No semantic correctness guarantee for filtered public corpus.',
        files={p.name:sha(p) for p in out.iterdir()})
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result['stats']),flush=True)

if __name__=='__main__':main()
