"""Full-source sampling, retaining the train-only vocabulary and lineage.

No prefix cap: deterministic 1/8 training sample across BOTH full raw files.
Held partitions use smallest digests over the full source (512 per kind).
Whole conversations only, at most512 tokens; never cut an assistant answer.
"""
import collections,heapq,hashlib,json,re,sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_conversation_v2 import ROOT,sha,clean,conversation,render,normalized,split_of,encode_dialogue
from scripts.curate_conversation import reject
PAST=re.compile(r'(?:我|最近).{0,20}(?:旅行笔记|整理旧物|去旅行|去了|去了趟|刚看完|早上跑步|早上散步|我过往经历|我小时候|我出生|我的童年|我的家人|我的朋友|我的同事)')

def main():
    source=ROOT/'runs/language-material';previous=ROOT/'runs/conversation-data-v3'
    out=ROOT/'runs/conversation-data-v4';out.mkdir(exist_ok=False)
    old=json.loads((previous/'manifest.json').read_text('utf-8'))
    for name in ('tokenizer.json','focus-tokens.json'):
        if sha(previous/name)!=old['files'][name]:raise ValueError('Previous artifact changed')
        (out/name).write_bytes((previous/name).read_bytes())
    tokenizer=Tokenizer.from_file(str(out/'tokenizer.json'));stats=collections.Counter();seen=set();sources={}
    held=collections.defaultdict(list)
    for kind,filename,provenance in [('pretrain','pretrain_t2t_mini.jsonl','pretrain-provenance.json'),('dialogue','sft_t2t_mini.jsonl','dialogue-provenance.json')]:
        path=source/filename;digest=sha(path);sources[filename]=digest
        if digest!=json.loads((source/provenance).read_text('utf-8'))['sha256']:raise ValueError('Raw file changed')
        position=0;size=path.stat().st_size
        with path.open('rb') as stream,(out/f'{kind}-train.jsonl').open('x',encoding='utf-8') as dest:
            for number,line in enumerate(stream):
                bucket=min(9,position*10//size);position+=len(line);stats[kind+'_scanned']+=1
                raw=json.loads(line);messages=None
                if kind=='pretrain':
                    text=clean(raw.get('text'));key=text
                    if text is None:stats[kind+'_filtered']+=1;continue
                else:
                    messages=conversation(raw)
                    if messages is None:stats[kind+'_filtered']+=1;continue
                    text=render(messages);key=messages[0]['content']
                digest=hashlib.sha256(normalized(text).encode()).hexdigest();split=split_of(key)
                if split=='train' and int(digest[:8],16)%8:continue
                record=dict(text=text,source_line=number,source_byte_bucket=bucket,sha256=digest,
                    group=hashlib.sha256(normalized(key)[:96].encode()).hexdigest())
                if messages is not None:record['messages']=messages
                why=reject(record,kind)
                if why:stats[kind+'_removed_'+why]+=1;continue
                if messages and any(PAST.search(m['content']) for m in messages if m['role']=='assistant'):
                    stats['dialogue_removed_more_invented_experience']+=1;continue
                if digest in seen:stats['exact_duplicates']+=1;continue
                seen.add(digest)
                if split=='train':
                    dest.write(json.dumps(record,ensure_ascii=False)+'\n');stats[kind+'_train_documents']+=1
                    stats[f'{kind}_train_bucket{bucket}']+=1
                else:
                    h=held[kind,split];item=(-int(digest,16),digest,record)
                    if len(h)<512:heapq.heappush(h,item)
                    elif item[0]>h[0][0]:heapq.heapreplace(h,item)
                if number and number%100000==0:print(json.dumps(dict(kind=kind,source_line=number,selected=stats[kind+'_train_documents'])),flush=True)
        for split in ('development','test'):
            with (out/f'{kind}-{split}.jsonl').open('x',encoding='utf-8') as dest:
                for _,_,record in sorted(held[kind,split],key=lambda r:r[1]):
                    dest.write(json.dumps(record,ensure_ascii=False)+'\n');stats[f'{kind}_{split}_documents']+=1
        print(json.dumps(dict(stage='full_scan',kind=kind,stats=stats)),flush=True)
    eos=tokenizer.token_to_id('<|eos|>');bos=tokenizer.token_to_id('<|bos|>')
    for split in ('train','development','test'):
        with (out/f'pretrain-{split}.jsonl').open(encoding='utf-8') as f,(out/f'pretrain-{split}.bin').open('xb') as binary:
            for line in f:
                seq=[bos]+tokenizer.encode(json.loads(line)['text']).ids+[eos]
                np.array(seq,dtype=np.uint16).tofile(binary);stats[f'pretrain_{split}_tokens']+=len(seq)
        with (out/f'dialogue-{split}.jsonl').open(encoding='utf-8') as f,(out/f'dialogue-{split}-tokens.jsonl').open('x',encoding='utf-8') as dest:
            for line in f:
                row=json.loads(line);value=encode_dialogue(tokenizer,row['messages'])
                if value is None:stats[f'dialogue_{split}_overlength']+=1;continue
                tokens,mask=value;dest.write(json.dumps(dict(tokens=tokens,mask=mask,group=row['group']))+'\n')
                stats[f'dialogue_{split}_encoded']+=1;stats[f'dialogue_{split}_targets']+=sum(mask)
                if len(tokens)<=257:stats[f'dialogue_{split}_under257']+=1
        print(json.dumps(dict(stage='encoded',split=split,stats=stats)),flush=True)
    manifest=dict(source_hashes=sources,previous_manifest_sha256=sha(previous/'manifest.json'),stats=stats,
        tokenizer_sha256=sha(out/'tokenizer.json'),vocabulary=tokenizer.get_vocab_size(),focus_segments=old['focus_segments'],
        source_code_sha256={name:sha(ROOT/name) for name in ('scripts/prepare_conversation_v4.py','scripts/prepare_conversation_v2.py','scripts/curate_conversation.py')},
        sampling='Full file scans, deterministic1/8 training sample with no prefix cap; lowest512 digests per heldout kind over full source.',
        split=old['split'],license=old['license'],
        limitations='Reuses v2 vocabulary learned on TRAIN only (including pieces from TRAIN docs excluded later). Regex filters incomplete; no semantic truth audit. Whole dialogues exceeding512 tokens excluded. New heldout differs from v3; compare both sets, not losses across different samples.',
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(stage='completed',stats=stats)),flush=True)

if __name__=='__main__':main()
