"""Reload paired candidate and test forced-memory spillover, research only."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel
from spora.research_memory_value_transform import MemoryValueTransform
from spora.research_vector_identity_select import select_identity
from scripts.deepseek_kernel_value_span_audit_20261117 import value_token_indices

SOURCE=ROOT/'runs/deepseek-kernel-paired-value-train-20261117'
OUT=ROOT/'runs/deepseek-kernel-paired-reload-audit-20261117.json'
EXTRA_MULTI_ENTITY=False
EXTRA_ORACLE_ROUTE=False


def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.perf_counter();torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    recorded=json.loads((SOURCE/'result.json').read_text(encoding='utf8'));candidate=SOURCE/'candidate.pt'
    if sha(candidate)!=recorded['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:raise ValueError('Input mismatch')
    ckpt=torch.load(candidate,map_location='cpu',weights_only=True)
    if ckpt['base_sha256']!=WEIGHT_SHA or ckpt['layer']!=5 or ckpt['head']!=0 or ckpt['value_transform'] is None:raise ValueError('Candidate mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
          torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    kernel=LearnedKernel().to('cuda');kernel.load_state_dict(ckpt['state']);kernel.eval()
    transform=MemoryValueTransform(128).to('cuda');transform.load_state_dict(ckpt['value_transform']);transform.eval()
    attn=model.model.layers[5].self_attn;captured={};active={'kv':None,'positions':None,'calls':0}
    def record(name):return lambda mod,args,out:captured.__setitem__(name,out.detach())
    def inject(mod,args):
        if active['kv'] is None:return None
        hidden=args[0];n=hidden.shape[1];raw=captured['q'].view(1,n,12,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=raw.dtype),active['positions'])
        rotated,_=apply_rotary_pos_emb(raw,raw,cos,sin)
        query=rotated[0,:6].float().reshape(-1,128);keys,values=active['kv']
        qf=F.softplus(kernel.qmap(F.layer_norm(query,(128,))))+1e-5
        kf=F.softplus(kernel.kmap(F.layer_norm(keys,(128,))))+1e-5
        pred=transform(qf@(kf.T@values)/(qf@kf.sum(0)).unsqueeze(-1))
        delta=pred.reshape(6,n,128).permute(1,0,2).reshape(1,n,768)
        active['calls']+=1
        return (torch.cat((hidden[:,:,:768]+delta.to(hidden.dtype),hidden[:,:,768:]),-1),)+args[1:]
    hooks=[getattr(attn,n+'_proj').register_forward_hook(record(n)) for n in ('q','k','v')]
    hooks.append(attn.o_proj.register_forward_pre_hook(inject))
    def make_state(name,value):
        ids=tok(f'{name}最喜欢的饮品是{value}。',return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
        active['kv']=None;model.model(input_ids=ids,use_cache=False);n=ids.shape[1]
        k=captured['k'].view(1,n,2,128).transpose(1,2);v=captured['v'].view(1,n,2,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),torch.arange(n,device='cuda').unsqueeze(0))
        _,k=apply_rotary_pos_emb(k,k,cos,sin)
        return k[0,0].float(),v[0,0].float()
    def generate(question,state):
        ids=closed_think_inputs(tok,question,'cuda')['input_ids'];start=ids.shape[1];active['calls']=0
        for _ in range(16):
            active['kv']=state;active['positions']=torch.arange(recorded['offset'],recorded['offset']+ids.shape[1],device='cuda').unsqueeze(0)
            nxt=model(input_ids=ids,position_ids=active['positions'],use_cache=False).logits[:,-1].argmax(-1).reshape(1,1)
            ids=torch.cat((ids,nxt),1)
            if nxt.item()==tok.eos_token_id:break
        result=ids[0,start:].tolist()
        return {'answer':tok.decode(result,skip_special_tokens=True).strip(),'token_ids':result,'injection_calls':active['calls']}
    try:
        with torch.inference_mode():
            rows=[]
            for old in recorded['held_responses']:
                state=make_state(old['name'],old['value'])
                output=generate(f"{old['name']}最喜欢的饮品是什么？请简洁回答。",state)
                rows.append({'name':old['name'],'value':old['value'],'same_tokens_as_training_process':output['token_ids']==old['token_ids'],**output})
            forced=make_state('小赵','可乐')
            questions=('小钱最喜欢的饮品是什么？请简洁回答。','三加四等于多少？','红灯亮时，行人应该怎么做？')
            challenges=[]
            for q in questions:
                off=generate(q,None);on=generate(q,forced)
                challenges.append({'question':q,'off':off,'forced_unrelated_memory':on,'same_tokens':off['token_ids']==on['token_ids']})
            multi=[]
            if EXTRA_MULTI_ENTITY:
                first=make_state('小赵','咖啡');second=make_state('小林','牛奶')
                merged=(torch.cat((first[0],second[0]),dim=0),torch.cat((first[1],second[1]),dim=0))
                for name,own,other in (('小赵',first,second),('小林',second,first)):
                    q=f'{name}最喜欢的饮品是什么？请简洁回答。'
                    for label,state in (('off',None),('own',own),('other',other),('merged',merged)):
                        multi.append({'name':name,'condition':label,**generate(q,state)})
                q='小钱最喜欢的饮品是什么？请简洁回答。'
                multi.append({'name':'小钱','condition':'merged',**generate(q,merged)})
            routed=[]
            if EXTRA_ORACLE_ROUTE:
                threshold_data=json.loads((ROOT/'runs/identity-embedding-similarity-20261117.json').read_text(encoding='utf8'))
                if threshold_data['base_sha256']!=WEIGHT_SHA:raise ValueError('Threshold base mismatch')
                threshold=threshold_data['threshold']
                def identity_vector(text,name):
                    encoded=tok(text,return_offsets_mapping=True,add_special_tokens=False)
                    positions=value_token_indices(text,name,encoded['offset_mapping'])
                    ids=torch.tensor([encoded['input_ids'][i] for i in positions],device='cuda')
                    return model.model.embed_tokens(ids).float().mean(0)
                supplied=(('小赵','咖啡'),('小林','牛奶'))
                record_vectors=torch.stack([identity_vector(f'{n}最喜欢的饮品是{v}。',n) for n,v in supplied])
                record_states=[make_state(n,v) for n,v in supplied]
                for name in ('小赵','小林','小钱','小林子'):
                    question=f'{name}最喜欢的饮品是什么？请简洁回答。'
                    decision=select_identity(identity_vector(question,name),record_vectors,threshold)
                    index=decision['index']
                    output=generate(question,record_states[index]) if index is not None else None
                    routed.append({'question':question,'oracle_subject':name,'decision':decision,'response':output})
    finally:
        for h in hooks:h.remove()
    if sha(candidate)!=recorded['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:raise ValueError('Protected input changed')
    result={'base_sha256':WEIGHT_SHA,'candidate_sha256':recorded['candidate_sha256'],'reload_rows':rows,
            'challenges':challenges,'multi_entity':multi,'oracle_routed':routed,'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Forced unrelated memory, not automatic routing or natural write.',
                           'Small authored set; responses capped at 16 tokens.',
                           'Regenerates state, not persisted individual recovery.',
                           'Unknown person is queried while explicitly supplying another person\'s state.',
                           'Merged states use independently encoded facts with positions restarting at zero; no automatic routing.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
