"""Counterfactual teacher-forced value signal; research data remains local."""
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
from scripts.deepseek_kernel_question_format_20261127 import selected_cases,build_facts,FACTS,DATA
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter

RUN=ROOT/'runs/research/residual-kv-writer-20261127-v1'
OUT=ROOT/'runs/research/residual-writer-value-signal-20261127-v1'
ALTERNATIVES=('足球','刘谦','红酒','周杰伦','程序员','披萨','摇滚','五月天')
FORCED_PREFIX='答案是：'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        paths={'base':MODEL/'model.safetensors','candidate':RUN/'candidate.pt','report':RUN/'result.json','data':DATA}
        hashes={k:sha(v) for k,v in paths.items()};report=json.loads(paths['report'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['candidate']!=report['candidate_sha256']:raise ValueError('source hash mismatch')
        payload=torch.load(paths['candidate'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('candidate base mismatch')
        _,values,questions=selected_cases(json.loads(DATA.read_text(encoding='utf8')));facts=build_facts(values)
        if any(a==b for a,b in zip(values,ALTERNATIVES)):raise ValueError('same counterfactual value')
        if any(not q.startswith('对话中') for q in questions):raise ValueError('unexpected question prefix')
        alt_facts=[form.format(value) for form,value in zip(FACTS,ALTERNATIVES)]
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        writer=ResidualWriter(model.config.num_hidden_layers).to('cuda').eval();writer.load_state_dict(payload['writer']);writer.requires_grad_(False)
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        with torch.no_grad():
            header=plain('')[:,:2];header_cache=tuple((k.detach(),v.detach()) for k,v in model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache())
            def state(f):
                ids=tok(f+'\n',add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
                source=tuple((k.float(),v.float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
                return writer(source)
            def score(q,memory,value):
                prompt=plain(q);suffix=prompt[:,2:];pref=tok.encode(FORCED_PREFIX,add_special_tokens=False)
                combined=tok.encode(FORCED_PREFIX+value,add_special_tokens=False)
                if combined[:len(pref)]!=pref or len(combined)<=len(pref):raise ValueError('target boundary mismatch')
                target=combined[len(pref):];answer=torch.tensor([pref+target],device='cuda');ids=torch.cat((suffix,answer),1)
                if memory is None:
                    logits=model(input_ids=torch.cat((header,ids),1),use_cache=False).logits.float();offset=2
                else:
                    entries=tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache));p=entries[0][0].shape[2]
                    logits=model(input_ids=ids,past_key_values=DynamicCache.from_legacy_cache(entries),attention_mask=torch.ones(1,p+ids.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+ids.shape[1],device='cuda'),use_cache=False).logits.float();offset=0
                startpos=offset+suffix.shape[1]+len(pref)-1;positions=logits[0,startpos:startpos+len(target)]
                labels=torch.tensor(target,device='cuda');logp=F.log_softmax(positions,-1).gather(1,labels[:,None]).squeeze(-1)
                return {'target_tokens':len(target),'first_logp':float(logp[0]),'sum_logp':float(logp.sum()),'mean_logp':float(logp.mean()),
                        'first_rank':int((positions[0]>positions[0,labels[0]]).sum())+1}
            rows=[];differences=[]
            for i,(value,alt,question) in enumerate(zip(values,ALTERNATIVES,questions)):
                orig=state(facts[i]);changed=state(alt_facts[i]);direct=question[len('对话中'):]+'请简洁回答。'
                for layer,((a,b),(c,d)) in enumerate(zip(orig,changed)):
                    for kind,x,y in [('key',a,c),('value',b,d)]:
                        x=x.reshape(-1).float();y=y.reshape(-1).float()
                        differences.append({'index':i,'layer':layer,'kind':kind,'cosine':float(F.cosine_similarity(x,y,dim=0)),
                                            'relative_change':float((x-y).norm()/x.norm().clamp_min(1e-9))})
                conditions=[('off',direct,None),('original_state',direct,orig),('alternate_state',direct,changed),
                            ('original_prompt',facts[i]+'\n'+direct,None),('alternate_prompt',alt_facts[i]+'\n'+direct,None)]
                for mode,q,mem in conditions:
                    a=score(q,mem,value);b=score(q,mem,alt)
                    rows.append({'index':i,'mode':mode,'original':a,'alternate':b,
                                 'sum_margin_original_minus_alternate':a['sum_logp']-b['sum_logp'],
                                 'mean_margin_original_minus_alternate':a['mean_logp']-b['mean_logp'],
                                 'first_margin_original_minus_alternate':a['first_logp']-b['first_logp']})
                save('progress.json',{'rows':rows,'differences':differences});print(json.dumps({'case':i+1,'rows':len(rows)}),flush=True)
        after={k:sha(v) for k,v in paths.items()};save('result.json',{'rows':rows,'state_differences':differences,
            'input_hashes':hashes,'input_hashes_after':after,'seconds':time.perf_counter()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Previously examined eight values and manually chosen alternatives; not independent.',
                           'Teacher forcing with explicit answer prefix and two choices is not free generation.',
                           'Manually supplied states, no autonomous memory routing or durable individual.',
                           'Different target token lengths affect scores; exact comparisons are exploratory.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
