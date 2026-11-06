"""Normalize public text, split before tokenizer training, and pack local arrays."""
import argparse,hashlib,json,re,sys,unicodedata,collections
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
SPECIAL=['<|pad|>','<|bos|>','<|eos|>','<|user|>','<|assistant|>']

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def normalized(text):return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).lower()

def split_of(key):
    # Group exact texts and shared leading passages together. This is not a
    # complete semantic near-duplicate detector; final chat needs own review.
    value=int(hashlib.sha256(normalized(key)[:96].encode()).hexdigest()[:8],16)%1000
    return 'development' if value<10 else ('test' if value<20 else 'train')

def clean(text):
    if not isinstance(text,str):return None
    text=unicodedata.normalize('NFC',text).strip()
    text=re.sub(r'<think>\s*</think>\s*','',text)
    if any(x in text.lower() for x in ('<think>','</think>','<tool_call>','<|','as an ai','我是chatgpt','我是一个ai','我是人工智能助手','我是由','作为一个ai','作为ai','我是minimind')):return None
    if '\ufffd' in text or len(text)<2 or len(text)>10000:return None
    if len(re.findall(r'[\u4e00-\u9fff]',text))<max(2,len(text)*.2):return None
    if re.search(r'(.{2,20})\1{4,}',text):return None
    return text

def conversation(row):
    items=row.get('conversations',[])
    if not isinstance(items,list) or len(items)<2:return None
    messages=[]
    for i,item in enumerate(items):
        if not isinstance(item,dict) or any(item.get(k) for k in ('tools','tool_calls')):return None
        role=item.get('role');text=clean(item.get('content'))
        if role!=('user' if i%2==0 else 'assistant') or text is None:return None
        messages.append(dict(role=role,content=text))
    if len(messages)%2:messages.pop()
    return messages if len(messages)>=2 else None

def render(messages):
    return ''.join(('<|user|>' if m['role']=='user' else '<|assistant|>')+m['content']+'<|eos|>' for m in messages)

def encode_dialogue(tokenizer,messages,limit=512):
    tokens=[tokenizer.token_to_id('<|bos|>')];mask=[False]
    for message in messages:
        role=tokenizer.token_to_id('<|'+message['role']+'|>')
        content=tokenizer.encode(message['content']).ids+[tokenizer.token_to_id('<|eos|>')]
        if len(tokens)+1+len(content)>limit:return None
        tokens.extend([role]+content);mask.extend([False]+[message['role']=='assistant']*len(content))
    return tokens,mask

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/conversation-data-v1');p.add_argument('--pretrain-limit',type=int,default=100000);p.add_argument('--dialogue-limit',type=int,default=100000)
    args=p.parse_args();out=ROOT/args.output;out.mkdir(exist_ok=False)
    material=ROOT/'runs/language-material';stats=collections.Counter();source_hashes={}
    for label,filename in [('pretrain','pretrain_t2t_mini.jsonl'),('dialogue','sft_t2t_mini.jsonl')]:
        prov=json.loads((material/f'{label}-provenance.json').read_text('utf-8'))
        source_hashes[filename]=sha(material/filename)
        if source_hashes[filename]!=prov['sha256']:raise ValueError('Unverified material')
    streams={(kind,split):(out/f'{kind}-{split}.jsonl').open('x',encoding='utf-8') for kind in ('pretrain','dialogue') for split in ('train','development','test')}
    seen=set()
    try:
        for kind,filename,limit in [('pretrain','pretrain_t2t_mini.jsonl',args.pretrain_limit),('dialogue','sft_t2t_mini.jsonl',args.dialogue_limit)]:
            count=0
            with (material/filename).open(encoding='utf-8') as f:
                for line_index,line in enumerate(f):
                    stats[kind+'_scanned']+=1
                    row=json.loads(line)
                    if kind=='pretrain':
                        text=clean(row.get('text'));messages=None
                        if text is None:stats[kind+'_filtered']+=1;continue
                        key=text
                    else:
                        messages=conversation(row)
                        if messages is None:stats[kind+'_filtered']+=1;continue
                        text=render(messages);key=messages[0]['content']
                        if sum(len(m['content']) for m in messages)>1400:stats['dialogue_long_filtered']+=1;continue
                    digest=hashlib.sha256(normalized(text).encode()).hexdigest()
                    if digest in seen:stats['exact_duplicates']+=1;continue
                    seen.add(digest);split=split_of(key)
                    if split!='train' and stats[kind+'_'+split]>=512:continue
                    record=dict(text=text,source_line=line_index,sha256=digest,group=hashlib.sha256(normalized(key)[:96].encode()).hexdigest())
                    if messages is not None:record['messages']=messages
                    streams[kind,split].write(json.dumps(record,ensure_ascii=False)+'\n');stats[kind+'_'+split]+=1
                    if split=='train':count+=1
                    if count>=limit:break
            print(json.dumps(dict(stage='normalized',kind=kind,stats=stats)),flush=True)
    finally:
        for stream in streams.values():stream.close()
    tokenizer=Tokenizer(models.BPE());tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)
    tokenizer.decoder=decoders.ByteLevel()
    def texts():
        for kind in ('pretrain','dialogue'):
            with (out/f'{kind}-train.jsonl').open(encoding='utf-8') as stream:
                for i,line in enumerate(stream):
                    if i>=40000:break
                    yield json.loads(line)['text']
    trainer=trainers.BpeTrainer(vocab_size=8192,min_frequency=3,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),show_progress=False)
    tokenizer.train_from_iterator(texts(),trainer=trainer);tokenizer.save(str(out/'tokenizer.json'))
    print(json.dumps(dict(stage='tokenizer',size=tokenizer.get_vocab_size())),flush=True)
    eos=tokenizer.token_to_id('<|eos|>');bos=tokenizer.token_to_id('<|bos|>')
    for split in ('train','development','test'):
        with (out/f'pretrain-{split}.bin').open('xb') as binary:
            with (out/f'pretrain-{split}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    seq=[bos]+tokenizer.encode(json.loads(line)['text']).ids+[eos]
                    np.array(seq,dtype=np.uint16).tofile(binary);stats['pretrain_'+split+'_tokens']+=len(seq)
        with (out/f'dialogue-{split}-tokens.jsonl').open('x',encoding='utf-8') as encoded:
            with (out/f'dialogue-{split}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line);value=encode_dialogue(tokenizer,row['messages'])
                    if value is None:stats['dialogue_'+split+'_token_length_filtered']+=1;continue
                    tokens,mask=value;encoded.write(json.dumps(dict(tokens=tokens,mask=mask,group=row['group']))+'\n')
                    stats['dialogue_'+split+'_encoded']+=1;stats['dialogue_'+split+'_targets']+=sum(mask)
        print(json.dumps(dict(stage='encoded',split=split,stats=stats)),flush=True)
    # Preserve the original three groups verbatim as genuine separate replay.
    sys.path.insert(0,str(ROOT/'scripts'))
    from chinese_curriculum_common import plan_data
    _,docs,_,_,_=plan_data();focus=[]
    for doc in docs:
        if doc['split']=='train' and doc['path'].startswith('focus-'):
            text=doc['text'];seq=[bos]+tokenizer.encode(text).ids+[eos]
            for start in range(0,len(seq)-1,255):focus.append(seq[start:start+257])
    (out/'focus-tokens.json').write_text(json.dumps(focus),encoding='utf-8')
    manifest=dict(source_hashes=source_hashes,stats=stats,tokenizer_sha256=sha(out/'tokenizer.json'),vocabulary=tokenizer.get_vocab_size(),
        source_code_sha256=sha(Path(__file__)),sampling='First accepted100k of each source; selection is order-biased, not random corpus sampling.',
        split='Grouped by normalized leading96 characters; 98/1/1 hash split, max512 heldout per kind. Exact text dedup across sources. Not exhaustive semantic near-dedup.',
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},focus_segments=len(focus),
        license='CC-BY-NC-4.0 as declared by corpus maintainer; local noncommercial research, source attribution in language-material provenance.')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
