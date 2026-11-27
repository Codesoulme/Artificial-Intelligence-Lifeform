"""Inspect writer attention and teacher-forced value response on synthetic dialogues."""
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
OUT=ROOT/'runs/research/dialogue-attention-signal-20261127-v1'
PREFIX='答案是：'

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
            def encode(text,fact,value):
                text+='\n';encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                ids=encoded['input_ids'].to('cuda');offset=encoded['offset_mapping'][0].tolist()
                pos=text.find(fact);valpos=text.find(value,pos)
                if pos<0 or valpos<0 or text.count(fact)!=1:raise ValueError('ambiguous spans')
                def mask(a,b):return torch.tensor([x<b and y>a for x,y in offset],device='cuda',dtype=torch.bool)
                fm=mask(pos,pos+len(fact));vm=mask(valpos,valpos+len(value))
                if not fm.any() or not vm.any():raise ValueError('empty evidence mask')
                state=tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
                return state,fm,vm
            def weights(writer,source,fm,vm):
                fact=[];value=[]
                for layer,(key,_) in enumerate(source):
                    k=key[0].float();score=torch.einsum('hnd,hsd->hsn',F.layer_norm(k,(128,)),writer.queries[layer])/128**0.5
                    attn=score.softmax(-1);fact.append(float(attn[...,fm].sum(-1).mean()));value.append(float(attn[...,vm].sum(-1).mean()))
                n=fm.numel();return {'tokens':n,'fact_tokens':int(fm.sum()),'value_tokens':int(vm.sum()),
                    'fact_mass_mean':sum(fact)/len(fact),'value_mass_mean':sum(value)/len(value),
                    'fact_uniform':int(fm.sum())/n,'value_uniform':int(vm.sum())/n,
                    'fact_mass_by_layer':fact,'value_mass_by_layer':value}
            def logits(q,memory):
                ids=plain(q)
                if not torch.equal(ids[:,:2],header):raise ValueError('header mismatch')
                pref=tok.encode(PREFIX,add_special_tokens=False);current=torch.cat((ids[:,2:],torch.tensor([pref],device='cuda')),1)
                pairs=tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache));p=pairs[0][0].shape[2]
                return model(input_ids=current,past_key_values=DynamicCache.from_legacy_cache(pairs),attention_mask=torch.ones(1,p+current.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+current.shape[1],device='cuda'),use_cache=False).logits[0,-1].float()
            rows=[];pairs=[]
            for relation,values in REL.items():
                a,b=values[:2];q=question('小陈',relation)
                pre=tok.encode(PREFIX,add_special_tokens=False);a_ids=tok.encode(PREFIX+a,add_special_tokens=False);b_ids=tok.encode(PREFIX+b,add_special_tokens=False)
                if a_ids[:len(pre)]!=pre or b_ids[:len(pre)]!=pre:raise ValueError('prefix token boundary')
                ta,tb=a_ids[len(pre)],b_ids[len(pre)]
                for form in ('short','dialogue'):
                    sources=[]
                    for value in (a,b):
                        case=('小陈',relation,value);fact=statement(*case)
                        sources.append(encode(fact if form=='short' else dialogue(case),fact,value))
                    for name,w in writers.items():
                        compressed=[]
                        for idx,(source,fm,vm) in enumerate(sources):
                            mem=w(source);compressed.append(mem);current=logits(q,mem);logp=F.log_softmax(current,-1)
                            rows.append({'relation':relation,'form':form,'candidate':name,'value_index':idx,
                                'attention':weights(w,source,fm,vm),'same_first_token':ta==tb,
                                'first_margin':float(logp[ta]-logp[tb])})
                        for layer,((ka,va),(kb,vb)) in enumerate(zip(*compressed)):
                            for kind,x,y in [('key',ka,kb),('value',va,vb)]:
                                pairs.append({'relation':relation,'form':form,'candidate':name,'layer':layer,'kind':kind,
                                    'relative_change':float((x-y).norm()/x.norm().clamp_min(1e-9))})
                save('progress.json',{'rows':rows,'differences':pairs});print(json.dumps({'relation':relation,'rows':len(rows)},ensure_ascii=True),flush=True)
        after={k:sha(p) for k,p in paths.items()}
        save('result.json',{'rows':rows,'differences':pairs,'input_hashes':hashes,'input_hashes_after':after,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Training-style synthetic dialogues and forced prefix with two choices.',
                'Attention weights alone do not establish causal contribution or semantic information retention.',
                'First-token preference does not establish complete free generation.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
