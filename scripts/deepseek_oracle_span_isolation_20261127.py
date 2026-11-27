"""Artificial evidence-span intervention on frozen dialogue KV states."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter
from scripts.deepseek_dialogue_distill_20261127 import REL,statement,question,dialogue

RUNS={'parent':ROOT/'runs/research/relation-coverage-20261127-v1',
      'distilled':ROOT/'runs/research/dialogue-distill-20261127-v2'}
OUT=ROOT/'runs/research/oracle-span-isolation-20261127-v1'; PREFIX='答案是：'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        paths={'base':MODEL/'model.safetensors'}
        for n,p in RUNS.items():paths[n]=p/'candidate.pt';paths[n+'_report']=p/'result.json'
        hashes={n:sha(p) for n,p in paths.items()}
        if hashes['base']!=WEIGHT_SHA:raise ValueError('wrong base')
        for n,p in RUNS.items():
            if hashes[n]!=json.loads((p/'result.json').read_text(encoding='utf8'))['candidate_sha256']:raise ValueError('wrong candidate')
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        writers={}
        for name in RUNS:
            payload=torch.load(paths[name],map_location='cpu',weights_only=True)
            if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('wrong candidate base')
            w=ResidualWriter(model.config.num_hidden_layers).to('cuda').eval().requires_grad_(False);w.load_state_dict(payload['writer']);writers[name]=w
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        with torch.no_grad():
            header=plain('')[:,:2];header_cache=tuple((k.detach(),v.detach()) for k,v in model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache())
            def encode(case):
                text=dialogue(case)+'\n';fact=statement(*case);value=case[2]
                item=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                ids=item['input_ids'].to('cuda');offsets=item['offset_mapping'][0].tolist();pos=text.find(fact);vpos=text.find(value,pos)
                if pos<0 or vpos<0 or text.count(fact)!=1:raise ValueError('ambiguous span')
                def mask(a,b):return torch.tensor([x<b and y>a for x,y in offsets],device='cuda',dtype=torch.bool)
                fm=mask(pos,pos+len(fact));vm=mask(vpos,vpos+len(value))
                if not fm.any() or not vm.any():raise ValueError('empty mask')
                source=tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
                return source,{'whole':torch.ones(ids.shape[1],device='cuda',dtype=torch.bool),'fact':fm,'value':vm}
            def paircache(memory):
                return tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache))
            def run(q,pairs,extra=None):
                ids=plain(q)
                if not torch.equal(ids[:,:2],header):raise ValueError('header mismatch')
                cur=ids[:,2:] if extra is None else torch.cat((ids[:,2:],torch.tensor([extra],device='cuda')),1)
                p=pairs[0][0].shape[2]
                return model(input_ids=cur,past_key_values=DynamicCache.from_legacy_cache(pairs),attention_mask=torch.ones(1,p+cur.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+cur.shape[1],device='cuda'),use_cache=False).logits[0,-1].float()
            def gen(q,pairs):
                tokens=[]
                for _ in range(24):
                    t=run(q,pairs,tokens if tokens else None).argmax(-1).item();tokens.append(t)
                    if t==tok.eos_token_id:break
                return {'text':tok.decode(tokens,skip_special_tokens=True).strip(),'tokens':tokens,'ended':tokens[-1]==tok.eos_token_id}
            rows=[]
            for relation,values in REL.items():
                a,b=values[:2];q=question('小陈',relation);pref=tok.encode(PREFIX,add_special_tokens=False)
                aa=tok.encode(PREFIX+a,add_special_tokens=False);bb=tok.encode(PREFIX+b,add_special_tokens=False)
                if aa[:len(pref)]!=pref or bb[:len(pref)]!=pref:raise ValueError('prefix boundary')
                ta,tb=aa[len(pref)],bb[len(pref)]
                for idx,value in enumerate((a,b)):
                    source,masks=encode(('小陈',relation,value))
                    for name,w in writers.items():
                        for span,mask in masks.items():
                            selected=tuple((k[:,:,mask,:],v[:,:,mask,:]) for k,v in source)
                            cache=paircache(w(selected));pred=F.log_softmax(run(q,cache,pref),-1)
                            result=gen(q,cache)
                            rows.append({'relation':relation,'candidate':name,'span':span,'value_index':idx,
                                'source_tokens':int(mask.sum()),'same_first_token':ta==tb,
                                'first_margin':float(pred[ta]-pred[tb]),'target_in_text':value in result['text'],**result})
                save('progress.json',{'rows':rows});print(json.dumps({'relation':relation,'rows':len(rows)},ensure_ascii=True),flush=True)
        after={k:sha(p) for k,p in paths.items()}
        save('result.json',{'rows':rows,'input_hashes':hashes,'input_hashes_after':after,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Oracle spans require the correct answer location; not autonomous retrieval.',
                'Original contextual KV retained but selected subset has changed attention normalization and placement.',
                'Training-style synthetic dialogues; small sample; string presence and first-token preference are screens.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
