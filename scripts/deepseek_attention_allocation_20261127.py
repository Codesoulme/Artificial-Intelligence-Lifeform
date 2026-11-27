"""Instrument raw historical KV attention in a frozen model; research only."""
import json, math, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

LAYERS=(0,5,12,19,27)
OUT=ROOT/'runs/research/attention-allocation-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def save(name,obj):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();weight=MODEL/'model.safetensors';before=sha(weight)
        if before!=WEIGHT_SHA:raise ValueError('base digest mismatch')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        if model.config.num_attention_heads!=12 or model.config.num_key_value_heads!=2:raise ValueError('unexpected head shape')
        captured={};active={'layers':set(),'state':None,'positions':None,'bias':0.,'collect':False,'stats':{}}
        def record(layer,name):return lambda mod,args,out:captured.__setitem__((layer,name),out.detach())
        def inject(layer):
            def forward(mod,args):
                if layer not in active['layers']:return None
                h=args[0];n=h.shape[1]
                q=captured[layer,'q'].view(1,n,12,128).transpose(1,2)
                k=captured[layer,'k'].view(1,n,2,128).transpose(1,2)
                v=captured[layer,'v'].view(1,n,2,128).transpose(1,2)
                cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=q.dtype),active['positions'])
                q,k=apply_rotary_pos_emb(q,k,cos,sin);q=q[0].float();k=k[0].float();v=v[0].float()
                mk,mv,value_positions=active['state'][layer];m=mk.shape[1]
                k=torch.cat((mk,k),dim=1).repeat_interleave(6,dim=0);v=torch.cat((mv,v),dim=1).repeat_interleave(6,dim=0)
                scores=q@k.transpose(1,2)/math.sqrt(128);scores[:,:,:m]+=active['bias']
                mask=torch.arange(n,device='cuda').unsqueeze(0)<=torch.arange(n,device='cuda').unsqueeze(1)
                mask=torch.cat((torch.ones(n,m,device='cuda',dtype=torch.bool),mask),1)
                weights=scores.masked_fill(~mask.unsqueeze(0),float('-inf')).softmax(-1)
                if active['collect']:
                    active['stats'][str(layer)]={'history_mass_by_head':weights[:,-1,:m].sum(-1).tolist(),
                        'value_mass_by_head':weights[:,-1,value_positions].sum(-1).tolist()}
                output=(weights@v).permute(1,0,2).reshape(1,n,1536).to(h.dtype)
                return (output,)+args[1:]
            return forward
        hooks=[]
        for layer in LAYERS:
            attn=model.model.layers[layer].self_attn
            hooks += [getattr(attn,name+'_proj').register_forward_hook(record(layer,name)) for name in ('q','k','v')]
            hooks.append(attn.o_proj.register_forward_pre_hook(inject(layer)))
        cases=[('小林',r,v) for r,vals in [('饮品',('豆浆','咖啡')),('水果',('苹果','香蕉'))] for v in vals]
        with torch.no_grad():
            states={}
            for name,relation,value in cases:
                fact=f'{name}最喜欢的{relation}是{value}。';enc=tok(fact,return_tensors='pt',add_special_tokens=False,return_offsets_mapping=True)
                offsets=enc.pop('offset_mapping')[0].tolist();pos=fact.index(value);value_positions=[i for i,(a,b) in enumerate(offsets) if a<pos+len(value) and b>pos]
                if not value_positions:raise ValueError('Missing value token span')
                ids=enc.input_ids.to('cuda');active['layers']=set();model.model(input_ids=ids,use_cache=False);n=ids.shape[1];per_layer={}
                for layer in LAYERS:
                    k=captured[layer,'k'].view(1,n,2,128).transpose(1,2);v=captured[layer,'v'].view(1,n,2,128).transpose(1,2)
                    cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),torch.arange(n,device='cuda').unsqueeze(0))
                    _,k=apply_rotary_pos_emb(k,k,cos,sin);per_layer[layer]=(k[0].float(),v[0].float(),value_positions)
                states[(name,relation,value)]=(fact,per_layer,n)
            offset=max(n for _,_,n in states.values());rows=[]
            modes=[('off',(),0),('layer5',(5,),0),('layer19',(19,),0),('five_layers',LAYERS,0),('five_layers_bias2',LAYERS,2),('fact_prompt',(),0)]
            for case in cases:
                name,relation,value=case;fact,kv,_=states[case];question=f'{name}最喜欢的{relation}是什么？请简洁回答。'
                for label,layers,bias in modes:
                    text=fact+'\n'+question if label=='fact_prompt' else question
                    ids=closed_think_inputs(tok,text,'cuda')['input_ids'];first=ids.shape[1];stats={}
                    for step in range(24):
                        active.update(layers=set(layers),state=kv,positions=torch.arange(offset,offset+ids.shape[1],device='cuda').unsqueeze(0),bias=bias,collect=step==0,stats={})
                        logits=model(input_ids=ids,position_ids=active['positions'],use_cache=False).logits
                        if step==0:stats=active['stats']
                        t=logits[:,-1].argmax(-1).reshape(1,1);ids=torch.cat((ids,t),1)
                        if t.item()==tok.eos_token_id:break
                    token=ids[0,first:].tolist();answer=tok.decode(token,skip_special_tokens=True).strip()
                    rows.append({'case':case,'mode':label,'text':answer,'target_in_text':value in answer,
                                 'ended':bool(token and token[-1]==tok.eos_token_id),'tokens':token,'first_step_attention':stats})
                save('progress.json',{'rows':rows});print(json.dumps({'case':case,'rows':len(rows)},ensure_ascii=True),flush=True)
        for hook in hooks:hook.remove()
        after=sha(weight);save('result.json',{'rows':rows,'layers':LAYERS,'offset':offset,'base_sha_before':before,'base_sha_after':after,
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Raw KV from fact alone and forced addresses; no autonomous persistent memory.',
                           'Four authored facts, fixed name and questions; first generation step attention only.',
                           'Explicit fact prompt is a language reference, not internal memory.',
                           'Manual attention computation is float32 then cast to model dtype.']})
        print(json.dumps({'complete':True,'unchanged':before==after,'seconds':time.perf_counter()-started}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
