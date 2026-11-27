"""Compare full chat-prefix KV and isolated fact KV at matched positions."""
import json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

OUT=ROOT/'runs/research/prefix-isolation-alignment-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();weight=MODEL/'model.safetensors';before=sha(weight)
        if before!=WEIGHT_SHA:raise ValueError('base digest mismatch')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        cases=[('小林',r,v) for r,values in [('饮品',('豆浆','咖啡')),('水果',('苹果','香蕉'))] for v in values]
        def get_cache(ids,positions=None):
            out=model(input_ids=ids,position_ids=positions,use_cache=True)
            return tuple((k.detach().clone(),v.detach().clone()) for k,v in out.past_key_values.to_legacy_cache())
        def stats(a,b):
            a=a.float().reshape(-1,128);b=b.float().reshape(-1,128)
            return {'cosine':float(F.cosine_similarity(a,b,dim=-1).mean()),'relative_rmse':float((a-b).norm()/a.norm().clamp_min(1e-9))}
        def gen(prefix_cache,suffix,span,isolated=None,replace_layers=()):
            items=[]
            for layer,(key,value) in enumerate(prefix_cache):
                key=key.clone();value=value.clone()
                if layer in replace_layers:
                    key[:,:,span[0]:span[1],:]=isolated[layer][0]
                    value[:,:,span[0]:span[1],:]=isolated[layer][1]
                items.append((key,value))
            cache=DynamicCache.from_legacy_cache(tuple(items));p=prefix_cache[0][0].shape[2];n=suffix.shape[1]
            mask=torch.ones((1,p+n),device='cuda',dtype=torch.long)
            output=model(input_ids=suffix,past_key_values=cache,attention_mask=mask,cache_position=torch.arange(p,p+n,device='cuda'),use_cache=True)
            logits=output.logits[:,-1].float();first=logits.detach().clone();cache=output.past_key_values;tokens=[]
            for step in range(24):
                t=logits.argmax(-1).reshape(1,1);tokens.append(t.item())
                if t.item()==tok.eos_token_id or step==23:break
                mask=torch.ones((1,p+n+step+1),device='cuda',dtype=torch.long)
                output=model(input_ids=t,past_key_values=cache,attention_mask=mask,cache_position=torch.tensor([p+n+step],device='cuda'),use_cache=True)
                logits=output.logits[:,-1].float();cache=output.past_key_values
            return first,tokens
        rows=[];measurements=[]
        with torch.no_grad():
            for case in cases:
                name,relation,value=case;fact=f'{name}最喜欢的{relation}是{value}。';question=f'{name}最喜欢的{relation}是什么？请简洁回答。'
                rendered=tok.apply_chat_template([{'role':'user','content':fact+'\n'+question}],tokenize=False,add_generation_prompt=True)+'</think>\n\n'
                enc=tok(rendered,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True);offsets=enc.pop('offset_mapping')[0].tolist();ids=enc.input_ids.to('cuda')
                fact_start=rendered.index(fact);fact_end=fact_start+len(fact);val_start=fact_start+fact.index(value);val_end=val_start+len(value)
                crossing=[(a,b) for a,b in offsets if a<fact_end<b]
                if any(rendered[fact_end:b].strip() for a,b in crossing):raise ValueError('Boundary includes question')
                end_char=max([fact_end]+[b for a,b in crossing]);first=min(i for i,(a,b) in enumerate(offsets) if b>fact_start)
                end=min(i for i,(a,b) in enumerate(offsets) if b>end_char)
                positions=[i-first for i,(a,b) in enumerate(offsets) if a<val_end and b>val_start]
                iso=tok(rendered[fact_start:end_char],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
                if not torch.equal(iso,ids[:,first:end]) or not positions:raise ValueError('Fact token alignment mismatch')
                full=get_cache(ids[:,:end]);zero=get_cache(iso);matched=get_cache(iso,torch.arange(first,end,device='cuda').unsqueeze(0));suffix=ids[:,end:]
                for layer,(key,value_tensor) in enumerate(full):
                    for name_mode,other in [('isolated_zero',zero),('isolated_matched',matched)]:
                        row={'case':case,'layer':layer,'condition':name_mode}
                        for label,original,candidate in [('key',key,other[layer][0]),('value',value_tensor,other[layer][1])]:
                            src=original[:,:,first:end,:];row[label+'_fact']=stats(src,candidate)
                            row[label+'_target']=stats(src[:,:,positions,:],candidate[:,:,positions,:])
                        measurements.append(row)
                base,btokens=gen(full,suffix,(first,end));modes=[('native',None,()),('zero_all',zero,tuple(range(len(full)))),
                    ('matched_all',matched,tuple(range(len(full)))),('zero_layer5',zero,(5,)),('matched_layer5',matched,(5,))]
                for mode,iso_cache,layers in modes:
                    logits,tokens=(base,btokens) if mode=='native' else gen(full,suffix,(first,end),iso_cache,layers)
                    text=tok.decode(tokens,skip_special_tokens=True).strip()
                    rows.append({'case':case,'mode':mode,'text':text,'tokens':tokens,'target_in_text':value in text,
                                 'ended':bool(tokens and tokens[-1]==tok.eos_token_id),
                                 'same_first_argmax':bool(torch.equal(base.argmax(-1),logits.argmax(-1))),
                                 'first_logit_max_abs_difference':float((logits-base).abs().max()),
                                 'prefix_tokens':end,'fact_span':[first,end],'target_positions':positions})
                save('progress.json',{'rows':rows,'measurements':measurements});print(json.dumps({'case':case,'rows':len(rows)},ensure_ascii=True),flush=True)
        after=sha(weight);save('result.json',{'rows':rows,'measurements':measurements,'base_sha_before':before,'base_sha_after':after,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Artificial fact is in human-authored chat prefix, no autonomous durable memory.',
                           'Small fixed dataset; replacement is an oracle intervention, not an online writer.',
                           'Cosine and relative error do not establish causal source of representation difference.']})
        print(json.dumps({'complete':True,'unchanged':before==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
