"""Noncommercial, local-only PerLTQA development dialogue compression probe."""
import hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter
from scripts.audit_perltqa_zh_20261127 import flatten

SRC=ROOT/'runs/research/perltqa-source-20261127'
RUN=ROOT/'runs/research/relation-coverage-20261127-v1'
OUT=ROOT/'runs/research/perltqa-dialogue-pilot-20261127-v1'
SALT='spora-perltqa-dialogue-pilot-20261127'

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        paths={'base':MODEL/'model.safetensors','candidate':RUN/'candidate.pt','report':RUN/'result.json',
               'split':SRC/'character-split.json','memory':SRC/'Dataset/zh/perltmem.json',
               'qa':SRC/'Dataset/zh/perltqa.json','manifest':SRC/'manifest.json'}
        hashes={k:sha(v) for k,v in paths.items()}
        report=json.loads(paths['report'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['candidate']!=report['candidate_sha256']:raise ValueError('Unexpected checkpoint')
        split=json.loads(paths['split'].read_text(encoding='utf8'));manifest=json.loads(paths['manifest'].read_text(encoding='utf8'))
        if split['source_commit']!=manifest['commit'] or hashes['memory']!=manifest['files']['Dataset/zh/perltmem.json']['sha256'] or hashes['qa']!=manifest['files']['Dataset/zh/perltqa.json']['sha256']:raise ValueError('Unexpected source')
        memory=json.loads(paths['memory'].read_text(encoding='utf8')); data=json.loads(paths['qa'].read_text(encoding='utf8'))
        mem={x['profile']['Protagonist']:x for x in memory};qa={k:v for x in data for k,v in x.items()}
        cases=[]
        for person in split['character_names']['development']:
            options=[]
            for entry in split['splits']['development']:
                if entry['character']!=person:continue
                key=entry['dialogue_key'];text='\n'.join(flatten(mem[person]['dialogues'][key]['contents']))
                if len(text)>500:continue
                for index,row in enumerate(qa[person]['dialogues'][key]):
                    answer=row['Answer'];question=row['Question']
                    if isinstance(answer,str) and isinstance(question,str) and 2<=len(answer)<=50 and answer in text and answer not in question:
                        order=hashlib.sha256((SALT+person+key+str(index)).encode('utf8')).digest()
                        options.append((order,key,index,text,question,answer))
            chosen=[];used=set()
            for _,key,index,text,question,answer in sorted(options):
                if key not in used:chosen.append({'person':person,'key':key,'index':index,'dialogue':text,'question':question,'answer':answer});used.add(key)
                if len(chosen)==2:break
            if len(chosen)!=2:raise ValueError('Insufficient eligible development cases')
            cases.extend(chosen)
        save('selection.json',{'cases':cases,'input_hashes':hashes,'salt':SALT})
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        payload=torch.load(paths['candidate'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('Wrong base')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        writer=ResidualWriter(model.config.num_hidden_layers).to('cuda').eval().requires_grad_(False);writer.load_state_dict(payload['writer'])
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        with torch.no_grad():
            header=plain('')[:,:2];header_cache=tuple((k.detach(),v.detach()) for k,v in model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache())
            def state(text):
                ids=tok(text+'\n',return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
                source=tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
                return writer(source),ids.shape[1]
            states=[state(c['dialogue']) for c in cases]
            def gen(q,memory):
                ids=plain(q)
                if not torch.equal(ids[:,:2],header):raise ValueError('Header mismatch')
                suffix=ids[:,2:];tokens=[]
                if memory is not None:
                    pairs=tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache));p=pairs[0][0].shape[2]
                for _ in range(48):
                    cur=torch.cat((suffix,torch.tensor([tokens],device='cuda')),1) if tokens else suffix
                    if memory is None:logits=model(input_ids=torch.cat((header,cur),1),use_cache=False).logits
                    else:logits=model(input_ids=cur,past_key_values=DynamicCache.from_legacy_cache(pairs),
                        attention_mask=torch.ones(1,p+cur.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+cur.shape[1],device='cuda'),use_cache=False).logits
                    t=logits[:,-1].argmax(-1).item();tokens.append(t)
                    if t==tok.eos_token_id:break
                return {'text':tok.decode(tokens,skip_special_tokens=True).strip(),'tokens':tokens,'ended':tokens[-1]==tok.eos_token_id}
            rows=[]
            for i,c in enumerate(cases):
                q=c['question']+'请简洁回答。';other=(i+2)%len(cases)
                for mode,prompt,source in [('off',q,None),('compressed',q,states[i][0]),
                    ('other_dialogue',q,states[other][0]),('full_text',c['dialogue']+'\n'+q,None)]:
                    answer=gen(prompt,source);rows.append({'index':i,'mode':mode,'answer':c['answer'],
                        'target_in_text':c['answer'] in answer['text'],**answer})
                save('progress.json',{'rows':rows});print(json.dumps({'case':i+1,'rows':len(rows)}),flush=True)
        after={k:sha(v) for k,v in paths.items()}
        save('result.json',{'rows':rows,'source_token_counts':[x[1] for x in states],
            'input_hashes':hashes,'input_hashes_after':after,'seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Noncommercial synthetic dialogue, selected for extractive answer; development cases.',
                'Artificial selected state, no autonomous extraction, addressing, persistence, or generic chat.',
                'String containment is a screening metric; unrelated state is not proven unanswerable.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-started}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
