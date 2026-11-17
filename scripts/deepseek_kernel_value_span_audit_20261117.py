"""Oracle span vs whole-fact state diagnostic; not autonomous memory."""
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
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel,OUT as INITIAL

OUT=ROOT/'runs/deepseek-kernel-value-span-audit-20261117.json'
TRAINED=ROOT/'runs/deepseek-kernel-counterfactual-train-20261117'
VALUES=('豆浆','咖啡');NAME='小赵'


def value_token_indices(text,value,offsets):
    start=text.index(value);end=start+len(value)
    found=[i for i,(a,b) in enumerate(offsets) if b>start and a<end]
    if not found or offsets[found[0]][0]!=start or offsets[found[-1]][1]!=end:
        raise ValueError('Value boundaries do not align with tokens')
    if ''.join(text[offsets[i][0]:offsets[i][1]] for i in found)!=value:
        raise ValueError('Value span not exactly covered')
    return found


def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.perf_counter();torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    sources={'initial':INITIAL,'trained':TRAINED}
    meta={name:json.loads((folder/'result.json').read_text(encoding='utf8')) for name,folder in sources.items()}
    if sha(MODEL/'model.safetensors')!=WEIGHT_SHA or any(sha(folder/'candidate.pt')!=meta[name]['candidate_sha256'] for name,folder in sources.items()):
        raise ValueError('Source mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
          torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    kernels={}
    for name,folder in sources.items():
        ckpt=torch.load(folder/'candidate.pt',map_location='cpu',weights_only=True)
        if ckpt['base_sha256']!=WEIGHT_SHA or ckpt['layer']!=5 or ckpt['head']!=0:raise ValueError('Wrong candidate')
        k=LearnedKernel().to('cuda');k.load_state_dict(ckpt['state']);k.eval();kernels[name]=k
    attn=model.model.layers[5].self_attn;captured={};active={'state':None,'kernel':None,'pos':None}
    def record(name):return lambda mod,args,out:captured.__setitem__(name,out.detach())
    def inject(mod,args):
        if active['state'] is None:return None
        hidden=args[0];n=hidden.shape[1];kernel=active['kernel']
        raw=captured['q'].view(1,n,12,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=raw.dtype),active['pos'])
        rotated,_=apply_rotary_pos_emb(raw,raw,cos,sin)
        query=rotated[0,:6].float().reshape(-1,128)
        qf=F.softplus(kernel.qmap(F.layer_norm(query,(128,))))+1e-5
        matrix,total=active['state'];read=qf@matrix/(qf@total).unsqueeze(-1)
        delta=read.reshape(6,n,128).permute(1,0,2).reshape(1,n,768)
        active['last_delta']=delta[0,-1].float()
        return (torch.cat((hidden[:,:,:768]+delta.to(hidden.dtype),hidden[:,:,768:]),-1),)+args[1:]
    hooks=[getattr(attn,n+'_proj').register_forward_hook(record(n)) for n in ('q','k','v')]
    hooks.append(attn.o_proj.register_forward_pre_hook(inject))
    def run(ids,state,kernel,offset):
        active.update(state=state,kernel=kernel,pos=torch.arange(offset,offset+ids.shape[1],device='cuda').unsqueeze(0))
        return model(input_ids=ids,position_ids=active['pos'],use_cache=False).logits[:,-1].float()
    def generate(ids,state,kernel,offset):
        start=ids.shape[1]
        for _ in range(16):
            token=run(ids,state,kernel,offset).argmax(-1).reshape(1,1);ids=torch.cat((ids,token),1)
            if token.item()==tok.eos_token_id:break
        tail=ids[0,start:].tolist()
        return {'answer':tok.decode(tail,skip_special_tokens=True).strip(),'token_ids':tail}
    try:
        with torch.inference_mode():
            facts=[f'{NAME}最喜欢的饮品是{value}。' for value in VALUES]
            encoded=[tok(fact,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False) for fact in facts]
            offset=max(x.input_ids.shape[1] for x in encoded);features=[]
            for fact,value,x in zip(facts,VALUES,encoded):
                positions=value_token_indices(fact,value,x.offset_mapping[0].tolist())
                ids=x.input_ids.to('cuda');n=ids.shape[1];active['state']=None
                model.model(input_ids=ids,use_cache=False)
                k=captured['k'].view(1,n,2,128).transpose(1,2);v=captured['v'].view(1,n,2,128).transpose(1,2)
                cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),torch.arange(n,device='cuda').unsqueeze(0))
                _,k=apply_rotary_pos_emb(k,k,cos,sin)
                features.append({'keys':k[0,0].float(),'values':v[0,0].float(),'positions':positions})
            qids=closed_think_inputs(tok,f'{NAME}最喜欢的饮品是什么？请简洁回答。','cuda')['input_ids']
            prefix=f'{NAME}最喜欢的饮品是';pre=tok.encode(prefix,add_special_tokens=False)
            labels=[]
            for value in VALUES:
                full=tok.encode(prefix+value,add_special_tokens=False)
                if full[:len(pre)]!=pre:raise ValueError('Answer prefix boundary mismatch')
                labels.append(full[len(pre)])
            forced=torch.cat((qids,torch.tensor([pre],device='cuda')),1)
            rows=[]
            for name,kernel in kernels.items():
                for mode in ('whole','value_only'):
                    for index,feat in enumerate(features):
                        pos=list(range(len(feat['keys']))) if mode=='whole' else feat['positions']
                        keys=feat['keys'][pos];vals=feat['values'][pos]
                        phi=F.softplus(kernel.kmap(F.layer_norm(keys,(128,))))+1e-5
                        state=(phi.T@vals,phi.sum(0))
                        logits=run(forced,state,kernel,offset)[0]
                        row={'candidate':name,'mode':mode,'fact_index':index,'value':VALUES[index],
                             'written_token_count':len(pos),'value_positions':feat['positions'],
                             'forced_target_logits':[float(logits[t]) for t in labels],
                             'forced_target_ranks':[int((logits>logits[t]).sum())+1 for t in labels],
                             'response':generate(qids.clone(),state,kernel,offset)}
                        rows.append(row)
                        print(json.dumps({'candidate':name,'mode':mode,'fact_index':index}),flush=True)
            off=generate(qids.clone(),None,None,offset)
    finally:
        for h in hooks:h.remove()
    if sha(MODEL/'model.safetensors')!=WEIGHT_SHA or any(sha(folder/'candidate.pt')!=meta[name]['candidate_sha256'] for name,folder in sources.items()):raise ValueError('Protected input changed')
    result={'base_sha256':WEIGHT_SHA,'candidate_sha256':{name:meta[name]['candidate_sha256'] for name in sources},
            'subject':NAME,'rows':rows,'no_memory':off,'seconds':time.perf_counter()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Value-only span is oracle supplied and removes subject context.',
                           'One person, two values, one layer/head and one question.',
                           'Fact values are encoded using the whole fact as preceding context.',
                           'No past KV cache, persistence, autonomous writes or unknown-question audit.',
                           'Forced token logits are not independent generated answers.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
