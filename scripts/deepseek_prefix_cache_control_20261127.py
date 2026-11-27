"""Compare exact prefix-cache continuation and value-position cache ablations."""
import json, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

OUT=ROOT/'runs/research/prefix-cache-control-20261127-v2'
LAYERS=(0,5,12,19,27)

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def save(name,data):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        weight=MODEL/'model.safetensors';before=sha(weight)
        if before!=WEIGHT_SHA:raise ValueError('Unexpected base digest')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        cases=[('小林',r,v) for r,values in [('饮品',('豆浆','咖啡')),('水果',('苹果','香蕉'))] for v in values]
        material={}
        with torch.no_grad():
            for case in cases:
                name,relation,value=case;fact=f'{name}最喜欢的{relation}是{value}。';q=f'{name}最喜欢的{relation}是什么？请简洁回答。'
                rendered=tok.apply_chat_template([{'role':'user','content':fact+'\n'+q}],tokenize=False,add_generation_prompt=True)+'</think>\n\n'
                encoded=tok(rendered,return_tensors='pt',add_special_tokens=False,return_offsets_mapping=True)
                offsets=encoded.pop('offset_mapping')[0].tolist();ids=encoded.input_ids.to('cuda')
                fact_start=rendered.index(fact);fact_end=fact_start+len(fact);value_start=fact_start+fact.index(value);value_end=value_start+len(value)
                value_pos=[i for i,(a,b) in enumerate(offsets) if a<value_end and b>value_start]
                crossing=[(a,b) for a,b in offsets if a<fact_end<b]
                if any(rendered[fact_end:b].strip() for a,b in crossing):raise ValueError('Split token contains question text')
                split_char=max([fact_end]+[b for a,b in crossing])
                suffix_positions=[i for i,(a,b) in enumerate(offsets) if b>split_char]
                split=min(suffix_positions)
                if not value_pos or max(value_pos)>=split:raise ValueError('Invalid value position')
                prefix=ids[:,:split];suffix=ids[:,split:]
                if not torch.equal(torch.cat((prefix,suffix),1),ids):raise ValueError('Split mismatch')
                result=model(input_ids=prefix,use_cache=True)
                legacy=tuple((k.detach().clone(),v.detach().clone()) for k,v in result.past_key_values.to_legacy_cache())
                material[case]={'ids':ids,'prefix':prefix,'suffix':suffix,'cache':legacy,'value_pos':value_pos,'fact':fact}
            def generate_full(ids):
                tokens=ids.clone();start=tokens.shape[1];first=None
                for _ in range(24):
                    logits=model(input_ids=tokens,use_cache=False).logits[:,-1].float()
                    if first is None:first=logits.detach().clone()
                    t=logits.argmax(-1).reshape(1,1);tokens=torch.cat((tokens,t),1)
                    if t.item()==tok.eos_token_id:break
                output=tokens[0,start:].tolist();return first,output
            def generate_cache(mat,zero_layers=()):
                p=mat['prefix'].shape[1];suffix=mat['suffix'];items=[]
                for index,(k,v) in enumerate(mat['cache']):
                    k=k.clone();v=v.clone()
                    if index in zero_layers:k[:,:,mat['value_pos'],:]=0;v[:,:,mat['value_pos'],:]=0
                    items.append((k,v))
                cache=DynamicCache.from_legacy_cache(tuple(items))
                n=suffix.shape[1];mask=torch.ones((1,p+n),device='cuda',dtype=torch.long)
                out=model(input_ids=suffix,past_key_values=cache,attention_mask=mask,
                          cache_position=torch.arange(p,p+n,device='cuda'),use_cache=True)
                logits=out.logits[:,-1].float();first=logits.detach().clone();cache=out.past_key_values;output=[]
                for step in range(24):
                    token=logits.argmax(-1).reshape(1,1);output.append(token.item())
                    if token.item()==tok.eos_token_id or step==23:break
                    mask=torch.ones((1,p+n+step+1),device='cuda',dtype=torch.long)
                    out=model(input_ids=token,past_key_values=cache,attention_mask=mask,
                              cache_position=torch.tensor([p+n+step],device='cuda'),use_cache=True)
                    cache=out.past_key_values;logits=out.logits[:,-1].float()
                return first,output
            def row(case,mode,first,output,base):
                text=tok.decode(output,skip_special_tokens=True).strip()
                return {'case':case,'mode':mode,'text':text,'tokens':output,'target_in_text':case[2] in text,
                        'ended':bool(output and output[-1]==tok.eos_token_id),
                        'max_abs_first_logit_delta':float((first-base).abs().max()),
                        'same_first_argmax':bool(torch.equal(first.argmax(-1),base.argmax(-1)))}
            rows=[]
            for case in cases:
                mat=material[case];base,base_tokens=generate_full(mat['ids']);rows.append(row(case,'full',base,base_tokens,base))
                conditions=[('cache',()),*((f'zero_layer_{l}',(l,)) for l in LAYERS),('zero_all',tuple(range(len(mat['cache']))))]
                for mode,zero_layers in conditions:
                    first,tokens=generate_cache(mat,zero_layers);rows.append(row(case,mode,first,tokens,base))
                other=next(x for x in cases if x[0]==case[0] and x[1]==case[1] and x[2]!=case[2]);alternative=material[other]
                if not torch.equal(mat['suffix'],alternative['suffix']) or mat['prefix'].shape!=alternative['prefix'].shape:
                    raise ValueError('Swap requires equal suffix and prefix lengths')
                swapped={**alternative,'suffix':mat['suffix']};first,tokens=generate_cache(swapped)
                swap=row(case,'swapped',first,tokens,base);swap['state_value']=other[2];swap['swapped_value_in_text']=other[2] in swap['text'];rows.append(swap)
                save('progress.json',{'rows':rows});print(json.dumps({'case':case,'rows':len(rows)},ensure_ascii=True),flush=True)
        after=sha(weight);result={'rows':rows,'base_sha_before':before,'base_sha_after':after,'layers':LAYERS,
            'splits':{str(c):{'prefix_tokens':m['prefix'].shape[1],'suffix_tokens':m['suffix'].shape[1],'value_positions':m['value_pos']} for c,m in material.items()},
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Chat prefix contains visible fact text; this is an inference cache, not autonomous durable memory.',
                           'Zeroed KV slots retain attention positions; this is an ablation, not exact deletion.',
                           'Four artificial cases and short greedy responses; matching strings do not imply generalization.']}
        save('result.json',result);print(json.dumps({'complete':True,'unchanged':before==after,'seconds':result['seconds']}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
