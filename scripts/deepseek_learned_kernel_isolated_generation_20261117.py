"""One-pair neural-state-only intervention; never modifies the base model."""
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel,OUT as TRAIN

OUT=ROOT/'runs/deepseek-learned-kernel-isolated-generation-v2-20261117.json'
FACTS=('小林最喜欢的饮品是豆浆。','小林最喜欢的饮品是咖啡。')
QUESTION='小林最喜欢的饮品是什么？请简洁回答。'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.perf_counter();torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    trained=json.loads((TRAIN/'result.json').read_text(encoding='utf8'));candidate=TRAIN/'candidate.pt'
    if sha(MODEL/'model.safetensors')!=WEIGHT_SHA or sha(candidate)!=trained['candidate_sha256']:
        raise ValueError('Input digest mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    ckpt=torch.load(candidate,map_location='cpu',weights_only=True)
    if ckpt['base_sha256']!=WEIGHT_SHA or ckpt['layer']!=5 or ckpt['head']!=0:
        raise ValueError('Candidate identity mismatch')
    kernel=LearnedKernel().to('cuda');kernel.load_state_dict(ckpt['state']);kernel.eval()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    attention=model.model.layers[5].self_attn; captured={};active={'state':None,'positions':None,'count':0}
    def record(name):
        return lambda mod,args,out:captured.__setitem__(name,out.detach())
    def inject(mod,args):
        if active['state'] is None:return None
        hidden=args[0];n=hidden.shape[1]
        raw=captured['q'].view(1,n,12,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=raw.dtype),active['positions'])
        rotated,_=apply_rotary_pos_emb(raw,raw,cos,sin)
        query=rotated[0,:6].float().reshape(-1,128)
        qf=F.softplus(kernel.qmap(F.layer_norm(query,(128,))))+1e-5
        matrix,total=active['state'];pred=qf@matrix/(qf@total).unsqueeze(-1)
        delta=pred.reshape(6,n,128).permute(1,0,2).reshape(1,n,768)
        modified=hidden.clone();modified[:,:,:768]+=delta.to(hidden.dtype)
        active['count']+=1
        return (modified,)+args[1:]
    hooks=[getattr(attention,n+'_proj').register_forward_hook(record(n)) for n in ('q','k','v')]
    hooks.append(attention.o_proj.register_forward_pre_hook(inject))
    def prompt_ids(question):return closed_think_inputs(tok,question,'cuda')['input_ids']
    def generate(ids,offset,state):
        active['state']=state;initial=ids.shape[1];active['count']=0
        eos=tok.eos_token_id
        for step in range(16):
            positions=torch.arange(offset,offset+ids.shape[1],device='cuda').unsqueeze(0)
            active['positions']=positions
            logits=model(input_ids=ids,position_ids=positions,use_cache=False).logits[:,-1].float()
            nxt=logits.argmax(-1).reshape(1,1);ids=torch.cat((ids,nxt),dim=1)
            if int(nxt.item())==eos:break
        tail=ids[0,initial:].tolist()
        return {'answer':tok.decode(tail,skip_special_tokens=True).strip(),'token_ids':tail,
                'memory_injection_calls':active['count']}
    try:
        with torch.inference_mode():
            fact_tokens=[tok(fact,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda') for fact in FACTS]
            offset=max(t.shape[1] for t in fact_tokens)
            states=[]
            for ids in fact_tokens:
                active['state']=None
                model.model(input_ids=ids,use_cache=False)
                n=ids.shape[1]
                k=captured['k'].view(1,n,2,128).transpose(1,2)
                v=captured['v'].view(1,n,2,128).transpose(1,2)
                pos=torch.arange(n,device='cuda').unsqueeze(0)
                cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),pos)
                _,k=apply_rotary_pos_emb(k,k,cos,sin)
                keys=k[0,0].float();values=v[0,0].float()
                phi=F.softplus(kernel.kmap(F.layer_norm(keys,(128,))))+1e-5
                states.append((phi.T@values,phi.sum(0)))
            # Only question tokens are passed to the decoder in the following
            # memory conditions. Actual facts are compressed into matrices.
            qids=prompt_ids(QUESTION)
            question_hash=hashlib.sha256(qids.cpu().numpy().tobytes()).hexdigest()
            state_hashes=[hashlib.sha256(b''.join(t.cpu().contiguous().numpy().tobytes() for t in state)).hexdigest()
                          for state in states]
            if state_hashes[0]==state_hashes[1]:raise ValueError('Counterfactual states are identical')
            off=generate(qids.clone(),offset,None)
            off_zero_offset=generate(qids.clone(),0,None)
            rows=[]
            for index,state in enumerate(states):
                result=generate(qids.clone(),offset,state)
                rows.append({'fact_index':index,'expected':FACTS[index],'response':result})
            references=[]
            for fact in FACTS:
                ids=prompt_ids(fact+'\n'+QUESTION)
                references.append(generate(ids,0,None))
    finally:
        for hook in hooks:hook.remove()
    if sha(candidate)!=trained['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:
        raise ValueError('Protected input changed')
    result={'candidate_sha256':trained['candidate_sha256'],'base_sha256':WEIGHT_SHA,'question':QUESTION,
            'question_input_sha256':question_hash,'question_offset':offset,'no_memory':off,'memory':rows,
            'no_memory_zero_offset':off_zero_offset,'state_sha256':state_hashes,
            'full_context_reference':references,'seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['One authored fact pair, one question, capped at 16 output tokens.',
                           'Only one KV head at one layer; direct residual gain 1 without trained gate.',
                           'Fact writer sees explicit short fact, not autonomous observation or identity routing.',
                           'Question decoder receives no past KV cache; matrix contains compressed key/value statistics.',
                           'No persisted individual or optimizer, no unrelated-query safety audit.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
