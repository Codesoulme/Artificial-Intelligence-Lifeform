"""Additional explicit identity/experience and repeated-answer filtering.

Keeps the train-only v2 tokenizer; removing examples does not import weights.
Documents and arrays are rebuilt in a new directory, original data retained.
"""
import json,re,hashlib,collections,sys
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_conversation_v2 import ROOT,sha,encode_dialogue
IDENTITY=re.compile(r'(你.{0,6}(模型|开发|训练|身份|创造|基于|数据集)|谁.{0,4}(开发|创造)|(?:我|本模型).{0,10}(?:由.{0,12}开发|名(?:为|叫)|基于.{0,8}模型)|minimind|jingyaogong|chatgpt|openai|通义千问|deepseek)',re.I)
LIVED=re.compile(r'(?:我|最近).{0,10}(?:和朋友|和家人|看了电影|去过|吃了|喝了|在赶进度|处理数据和优化|毛绒玩具|我出生|我小时候)')

def reject(row,kind):
    text=row['text']
    if IDENTITY.search(text):return 'identity'
    if kind=='dialogue':
        answers=[m['content'] for m in row['messages'] if m['role']=='assistant']
        if any(LIVED.search(x) for x in answers):return 'invented_experience'
        if len(set(answers))<len(answers):return 'duplicate_answers'
        suffixes=[re.sub(r'\s+','',x)[-16:] for x in answers]
        if len(suffixes)>=3 and max(collections.Counter(suffixes).values())>=3:return 'repeated_closing'
    return None

def main():
    source=ROOT/'runs/conversation-data-v2';out=ROOT/'runs/conversation-data-v3';out.mkdir(exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text('utf-8'))
    for name in manifest['files']:
        if sha(source/name)!=manifest['files'][name]:raise ValueError('Source artifact changed')
    tokenizer=Tokenizer.from_file(str(source/'tokenizer.json'));(out/'tokenizer.json').write_bytes((source/'tokenizer.json').read_bytes())
    (out/'focus-tokens.json').write_bytes((source/'focus-tokens.json').read_bytes())
    eos=tokenizer.token_to_id('<|eos|>');bos=tokenizer.token_to_id('<|bos|>');stats=collections.Counter()
    for split in ('train','development','test'):
        for kind in ('pretrain','dialogue'):
            with (source/f'{kind}-{split}.jsonl').open(encoding='utf-8') as f,(out/f'{kind}-{split}.jsonl').open('x',encoding='utf-8') as dest:
                for line in f:
                    row=json.loads(line);why=reject(row,kind)
                    if why:stats[f'{kind}_{split}_removed_{why}']+=1;continue
                    dest.write(line);stats[f'{kind}_{split}_documents']+=1
        with (out/f'pretrain-{split}.bin').open('xb') as binary,(out/f'pretrain-{split}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                seq=[bos]+tokenizer.encode(json.loads(line)['text']).ids+[eos];np.array(seq,dtype=np.uint16).tofile(binary)
                stats[f'pretrain_{split}_tokens']+=len(seq)
        with (out/f'dialogue-{split}-tokens.jsonl').open('x',encoding='utf-8') as dest,(out/f'dialogue-{split}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);value=encode_dialogue(tokenizer,row['messages'])
                if value is None:stats[f'dialogue_{split}_overlength']+=1;continue
                tokens,mask=value;dest.write(json.dumps(dict(tokens=tokens,mask=mask,group=row['group']))+'\n')
                stats[f'dialogue_{split}_encoded']+=1;stats[f'dialogue_{split}_targets']+=sum(mask)
        print(json.dumps(dict(split=split,stats=stats)),flush=True)
    result=dict(source_manifest_sha256=sha(source/'manifest.json'),source_hashes=manifest['source_hashes'],stats=stats,
        tokenizer_sha256=sha(out/'tokenizer.json'),vocabulary=tokenizer.get_vocab_size(),source_code_sha256=sha(Path(__file__)),
        files={p.name:sha(p) for p in out.iterdir()},focus_segments=manifest['focus_segments'],license=manifest['license'],
        split=manifest['split'],sampling=manifest['sampling'],
        limitations='Regex filters are imperfect, factual correctness not established. Reuses v2 train-only vocabulary, including word pieces from subsequently filtered TRAIN documents; no heldout vocabulary fitting. No pretrained weights.')
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
