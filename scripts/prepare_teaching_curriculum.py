"""Prepare a new training-only text curriculum with held-input exclusions."""
import collections,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import ROOT,sha,normalized,encode_dialogue,render
from scripts.curate_conversation import IDENTITY
from scripts.audit_conversation_data import signatures

def main():
    material=ROOT/'runs/alpaca-material';data=ROOT/'runs/conversation-data-v4';out=ROOT/'runs/teaching-curriculum-v2'
    out.mkdir(exist_ok=False);provenance=json.loads((material/'provenance.json').read_text('utf-8'))
    source=material/'alpaca_gpt4_data_zh.json'
    if sha(source)!=provenance['sha256']:raise ValueError('Unverified source')
    tok=Tokenizer.from_file(str(data/'tokenizer.json'));stats=collections.Counter();groups=set();digests=set();held=[];index=collections.defaultdict(list);protected_inputs=set()
    for directory in ('conversation-data-v3','conversation-data-v4'):
        for split in ('development','test'):
            for kind in ('pretrain','dialogue'):
                with (ROOT/'runs'/directory/f'{kind}-{split}.jsonl').open(encoding='utf-8') as f:
                    for line in f:
                        row=json.loads(line);groups.add(row['group']);digests.add(row['sha256']);sigs=signatures(row['text']);n=len(held);held.append(len(sigs))
                        for sig in sigs:index[sig].append(n)
                        if kind=='dialogue':
                            for m in row['messages']:
                                if m['role']=='user':protected_inputs.add(normalized(m['content']))
    # Explicit development scenarios remain development; their exact user
    # inputs and monitored prompts are excluded from new teaching material.
    dev=json.loads((ROOT/'data/conversation-development.json').read_text('utf-8'))
    protected_inputs.update(normalized(t) for c in dev['scenarios'] for t in c['turns'])
    from scripts.train_conversation_v2 import PROMPTS
    protected_inputs.update(map(normalized,PROMPTS))
    seen=set();audit=[]
    def isolated(messages,text):
        key=messages[0]['content'];group=hashlib.sha256(normalized(key)[:96].encode()).hexdigest();digest=hashlib.sha256(normalized(text).encode()).hexdigest()
        if group in groups or digest in digests:return False,group,digest,'held_group_or_exact'
        if any(normalized(m['content']) in protected_inputs for m in messages if m['role']=='user'):return False,group,digest,'held_question'
        sigs=signatures(text);counts=collections.Counter(i for s in sigs for i in index.get(s,()))
        if any(min(len(sigs),held[i])>=8 and n/min(len(sigs),held[i])>=.8 for i,n in counts.items()):return False,group,digest,'held_lexical_overlap'
        return True,group,digest,None
    def write_record(messages,tag,streams):
        text=render(messages);ok,group,digest,reason=isolated(messages,text)
        if not ok:stats[tag+'_removed_'+reason]+=1;audit.append(dict(source=tag,sha256=digest,reason=reason));return
        if digest in seen:stats[tag+'_duplicate']+=1;return
        value=encode_dialogue(tok,messages,limit=512)
        if value is None:stats[tag+'_overlength']+=1;return
        seen.add(digest);tokens,mask=value
        streams[0].write(json.dumps(dict(tokens=tokens,mask=mask,group=group))+'\n')
        streams[1].write(json.dumps(dict(messages=messages,text=text,group=group,sha256=digest,source=tag),ensure_ascii=False)+'\n')
        stats[tag+'_dialogues']+=1;stats[tag+'_targets']+=sum(mask);stats[tag+'_turns']+=len(messages)//2
    with (out/'teacher-tokens.jsonl').open('x',encoding='utf-8') as encoded,(out/'teacher.jsonl').open('x',encoding='utf-8') as raw:
        for row in json.loads(source.read_text('utf-8')):
            instruction=row['instruction'].strip();extra=row['input'].strip();answer=row['output'].strip()
            prompt=instruction+('\n'+extra if extra else '')
            if not prompt or not answer:stats['empty']+=1;continue
            if IDENTITY.search(prompt+'\n'+answer) or re.search(r'作为.{0,6}(?:AI|语言模型|人工智能)|OpenAI|GPT[- ]?[34]',answer,re.I):stats['identity']+=1;continue
            if any(x in prompt+answer for x in ('<|','\ufffd','```','<script','<html','<think>')):stats['code_or_control']+=1;continue
            if len(answer)>500 or len(prompt)>1000:stats['long_text']+=1;continue
            han=sum('\u4e00'<=c<='\u9fff' for c in answer);latin=sum('a'<=c.lower()<='z' for c in answer)
            if han<2 or han<.85*(han+latin):stats['non_chinese']+=1;continue
            if re.search(r'(.{2,20})\1{3,}',answer):stats['repetition']+=1;continue
            if re.search(r'(?:我|我们).{0,8}(?:出生|毕业|小时候|童年|我的家人|我曾经)',answer):stats['possible_identity']+=1;continue
            write_record([dict(role='user',content=prompt),dict(role='assistant',content=answer)],'teacher',(encoded,raw))
    authored=ROOT/'data/chinese-foundations.json';teaching=json.loads(authored.read_text('utf-8'))
    with (out/'foundations-tokens.jsonl').open('x',encoding='utf-8') as encoded,(out/'foundations.jsonl').open('x',encoding='utf-8') as raw:
        for dialogue in teaching['dialogues']:
            messages=[dict(role=role,content=s) for pair in dialogue['turns'] for role,s in zip(('user','assistant'),pair)]
            write_record(messages,'foundations',(encoded,raw))
    manifest=dict(source_provenance=provenance,authored_sha256=sha(authored),tokenizer_sha256=sha(data/'tokenizer.json'),
        source_sha256=sha(Path(__file__)),stats=stats,excluded_overlaps=audit,
        held_manifests={directory:sha(ROOT/'runs'/directory/'manifest.json') for directory in ('conversation-data-v3','conversation-data-v4')},
        development_questions_sha256=sha(ROOT/'data/conversation-development.json'),
        limitations='Training-only synthetic source, not independent truth verification. Excludes exact user prompts and finite lexical overlaps with held partitions; semantic paraphrases can remain. Authored examples are teaching, never final evaluation. No new tokenizer fitting.',
        files={p.name:sha(p) for p in out.iterdir()})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
