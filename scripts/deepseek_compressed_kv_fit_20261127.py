"""Per-fact optimized KV slots; tiny research probe, not an online memory writer."""
import json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_addressed_read import closed_think_inputs

OUT=ROOT/'runs/research/compressed-kv-fit-20261127-v1'
SEED,STEPS,LR=112728,32,0.01

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def save(name,data):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.manual_seed(SEED);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        weight=MODEL/'model.safetensors';before=sha(weight)
        if before!=WEIGHT_SHA:raise ValueError('Unexpected base digest')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        cases=[('小林',r,v) for r,vals in [('饮品',('豆浆','咖啡')),('水果',('苹果','香蕉'))] for v in vals]
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        # Taking only the leading special tokens provides no fact or answer text.
        template_ids=plain('')
        # All facts start after the two chat-header tokens in the matched control.
        header_ids=template_ids[:,:2]
        with torch.no_grad():
            header_cache=model(input_ids=header_ids,use_cache=True).past_key_values.to_legacy_cache()
            header_cache=tuple((k.detach().clone(),v.detach().clone()) for k,v in header_cache)
        def prompts(case):
            n,r,v=case;return f'{n}最喜欢的{r}是{v}。',f'{n}最喜欢的{r}是什么？请简洁回答。',f'{n}最爱喝什么？' if r=='饮品' else f'{n}最爱吃什么水果？'
        def suffix(q):
            ids=plain(q)
            if not torch.equal(ids[:,:2],header_ids):raise ValueError('Template header changed')
            return ids[:,2:]
        def make_slots(fact):
            ids=tok(fact+'\n',return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
            with torch.no_grad():cache=model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache()
            positions=[0,ids.shape[1]-1]
            keys=nn.ParameterList([nn.Parameter(k[:,:,positions,:].detach().float().clone()) for k,v in cache])
            vals=nn.ParameterList([nn.Parameter(v[:,:,positions,:].detach().float().clone()) for k,v in cache])
            return keys,vals,ids.shape[1]
        def run(q,keys,vals,answer_ids=None):
            seq=suffix(q);input_ids=seq if answer_ids is None else torch.cat((seq,answer_ids),1)
            entries=[]
            for layer,(hk,hv) in enumerate(header_cache):
                entries.append((torch.cat((hk,keys[layer].to(hk.dtype)),2),torch.cat((hv,vals[layer].to(hv.dtype)),2)))
            cache=DynamicCache.from_legacy_cache(tuple(entries));p=entries[0][0].shape[2]
            output=model(input_ids=input_ids,past_key_values=cache,attention_mask=torch.ones(1,p+input_ids.shape[1],device='cuda',dtype=torch.long),
                         cache_position=torch.arange(p,p+input_ids.shape[1],device='cuda'),use_cache=False)
            return output.logits.float(),seq.shape[1]
        def generate(q,keys,vals):
            seq=suffix(q);added=[]
            with torch.no_grad():
                for _ in range(24):
                    logits,_=run(q,keys,vals,torch.tensor([added],device='cuda',dtype=torch.long) if added else None)
                    token=logits[:,-1].argmax(-1).item();added.append(token)
                    if token==tok.eos_token_id:break
            return {'text':tok.decode(added,skip_special_tokens=True).strip(),'tokens':added,'ended':bool(added and added[-1]==tok.eos_token_id)}
        def no_memory(q):
            seq=plain(q);start=seq.shape[1]
            with torch.no_grad():
                for _ in range(24):
                    t=model(input_ids=seq,use_cache=False).logits[:,-1].argmax(-1).reshape(1,1);seq=torch.cat((seq,t),1)
                    if t.item()==tok.eos_token_id:break
            ids=seq[0,start:].tolist();return {'text':tok.decode(ids,skip_special_tokens=True).strip(),'tokens':ids,'ended':bool(ids and ids[-1]==tok.eos_token_id)}
        rows=[];traces=[];trained={};progress={}
        for case in cases:
            fact,q,held=prompts(case);keys,vals,fact_tokens=make_slots(fact)
            baseline=[]
            for label,question in [('train_question',q),('held_wording',held)]:
                answer=generate(question,keys,vals);baseline.append({'case':case,'phase':'before','condition':label,'question':question,'target_in_text':case[2] in answer['text'],**answer})
            rows.extend(baseline);save('progress.json',{'rows':rows,'traces':traces})
            answer_ids=torch.tensor([tok.encode(fact,add_special_tokens=False)+[tok.eos_token_id]],device='cuda')
            teacher_prompt=plain(fact+'\n'+q);teacher_ids=torch.cat((teacher_prompt,answer_ids),1)
            with torch.no_grad():
                logits=model(input_ids=teacher_ids,use_cache=False).logits[:,teacher_prompt.shape[1]-1:-1].float()
                teacher_prob=F.softmax(logits,dim=-1).detach()
            opt=torch.optim.AdamW(list(keys)+list(vals),lr=LR,weight_decay=0)
            case_trace=[]
            for step in range(STEPS):
                opt.zero_grad(set_to_none=True);out,plen=run(q,keys,vals,answer_ids)
                student=out[:,plen-1:-1];ce=F.cross_entropy(student.reshape(-1,student.shape[-1]),answer_ids.reshape(-1))
                kl=F.kl_div(F.log_softmax(student,dim=-1),teacher_prob,reduction='batchmean')/answer_ids.shape[1]
                loss=ce+0.5*kl
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(list(keys)+list(vals),1.0,error_if_nonfinite=True));opt.step()
                row={'step':step+1,'ce':float(ce.detach()),'kl':float(kl.detach()),'gradient_norm':grad};case_trace.append(row)
                if step in (0,7,15,31):print(json.dumps({'case':case,**row},ensure_ascii=True),flush=True)
            traces.append({'case':case,'steps':case_trace,'fact_token_count':fact_tokens})
            ckpt={'case':case,'keys':[x.detach().cpu() for x in keys],'values':[x.detach().cpu() for x in vals],
                  'seed':SEED,'base_sha256':WEIGHT_SHA,'slots_per_layer':2}
            path=OUT/('state-'+str(len(trained))+'.pt');torch.save(ckpt,path)
            trained[case]=(keys,vals,path)
            for label,question in [('train_question',q),('held_wording',held)]:
                answer=generate(question,keys,vals);rows.append({'case':case,'phase':'after','condition':label,'question':question,'target_in_text':case[2] in answer['text'],**answer})
            save('progress.json',{'rows':rows,'traces':traces})
        # Cross-fact swaps use the exact same question but another trained state.
        for case in cases:
            _,q,_=prompts(case);same=next(c for c in cases if c[1]==case[1] and c[2]!=case[2]);wrong=next(c for c in cases if c[1]!=case[1])
            for label,source in [('same_relation_other_value',same),('wrong_relation',wrong)]:
                k,v,_=trained[source];answer=generate(q,k,v)
                rows.append({'case':case,'phase':'after','condition':label,'state_case':source,'question':q,
                             'target_in_text':case[2] in answer['text'],'state_value_in_text':source[2] in answer['text'],**answer})
            save('progress.json',{'rows':rows,'traces':traces})
        k,v,_=trained[cases[0]]
        for q in ('请计算三加四。','请说一句鼓励人的话。','小林的职业是什么？请简洁回答。'):
            for mode in ('off','forced_unrelated'):
                answer=no_memory(q) if mode=='off' else generate(q,k,v)
                rows.append({'phase':'after','condition':mode,'question':q,**answer})
            save('progress.json',{'rows':rows,'traces':traces})
        after=sha(weight)
        save('result.json',{'rows':rows,'traces':traces,'steps_per_fact':STEPS,'learning_rate':LR,'seed':SEED,
            'base_sha_before':before,'base_sha_after':after,'states':[{'case':case,'path':str(p),'sha256':sha(p),
            'bytes':p.stat().st_size,'tensor_bytes':sum(t.numel()*t.element_size() for t in list(k)+list(v))} for case,(k,v,p) in trained.items()],
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Each manually specified fact is individually optimized using its answer; not an autonomous writer.',
                           'Four earlier-seen synthetic facts, one name and fixed relation templates; no independent acceptance.',
                           'Teacher sees facts in prompt; candidate states scale with number of facts.',
                           'No persistence of formal individual, automatic selection, correction, or crash recovery.']})
        print(json.dumps({'complete':True,'unchanged':before==after,'seconds':time.perf_counter()-started}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
