"""Oracle historical KV versus compressed-kernel read; not a persistent individual."""
import json, math, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel
from spora.research_memory_value_transform import MemoryValueTransform

SOURCE=ROOT/'runs/deepseek-kernel-full-answer-train-20261117'
OUT=ROOT/'runs/research/native-attention-oracle-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    def save(name,obj):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(OUT/name)
    try:
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        paths={'base':MODEL/'model.safetensors','source':SOURCE/'candidate.pt','report':SOURCE/'result.json'}
        hashes={k:sha(v) for k,v in paths.items()}; report=json.loads(paths['report'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['source']!=report['candidate_sha256']:raise ValueError('source mismatch')
        payload=torch.load(paths['source'],map_location='cpu',weights_only=True)
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        kernel=LearnedKernel().to('cuda').eval();kernel.load_state_dict(payload['state']);kernel.requires_grad_(False)
        transform=MemoryValueTransform(128).to('cuda').eval();transform.load_state_dict(payload['value_transform']);transform.requires_grad_(False)
        attn=model.model.layers[5].self_attn; captured={};active={'kv':None,'pos':None,'mode':'off'}
        def record(name):return lambda mod,args,out:captured.__setitem__(name,out.detach())
        def hook(mod,args):
            mode=active['mode']
            if mode=='off':return None
            h=args[0];n=h.shape[1];raw=captured['q'].view(1,n,12,128).transpose(1,2)
            k=captured['k'].view(1,n,2,128).transpose(1,2);v=captured['v'].view(1,n,2,128).transpose(1,2)
            cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=raw.dtype),active['pos'])
            q,k=apply_rotary_pos_emb(raw,k,cos,sin);q=q[0,:6].float();k=k[0,0].float();v=v[0,0].float()
            if mode=='kernel':
                mk,mv=active['kv'];flat=q.reshape(-1,128)
                qf=F.softplus(kernel.qmap(F.layer_norm(flat,(128,))))+1e-5
                kf=F.softplus(kernel.kmap(F.layer_norm(mk,(128,))))+1e-5
                pred=transform(qf@(kf.T@mv)/(qf@kf.sum(0)).unsqueeze(-1))
                delta=pred.reshape(6,n,128).permute(1,0,2).reshape(1,n,768)
                return (torch.cat((h[:,:,:768]+delta.to(h.dtype),h[:,:,768:]),-1),)+args[1:]
            m=0
            if mode=='oracle':
                mk,mv=active['kv'];m=mk.shape[0];k=torch.cat((mk,k));v=torch.cat((mv,v))
            scores=q@k.T/math.sqrt(128)
            mask=torch.arange(n,device='cuda').unsqueeze(0)<=torch.arange(n,device='cuda').unsqueeze(1)
            if m:mask=torch.cat((torch.ones(n,m,device='cuda',dtype=torch.bool),mask),dim=1)
            scores=scores.masked_fill(~mask.unsqueeze(0),float('-inf'))
            output=(scores.softmax(-1)@v).permute(1,0,2).reshape(1,n,768).to(h.dtype)
            return (torch.cat((output,h[:,:,768:]),-1),)+args[1:]
        hooks=[getattr(attn,n+'_proj').register_forward_hook(record(n)) for n in ('q','k','v')]
        hooks.append(attn.o_proj.register_forward_pre_hook(hook))
        def state(fact):
            ids=tok(fact,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda');active['mode']='off'
            model.model(input_ids=ids,use_cache=False);n=ids.shape[1]
            k=captured['k'].view(1,n,2,128).transpose(1,2);v=captured['v'].view(1,n,2,128).transpose(1,2)
            cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),torch.arange(n,device='cuda').unsqueeze(0))
            _,k=apply_rotary_pos_emb(k,k,cos,sin);return k[0,0].float(),v[0,0].float()
        cases=[(name,rel,value) for name in ('小林','小赵') for rel,values in [('饮品',('豆浆','咖啡')),('水果',('苹果','香蕉'))] for value in values]
        with torch.no_grad():
            states={c:state(f'{c[0]}最喜欢的{c[1]}是{c[2]}。') for c in cases}
            offset=max(report['offset'],max(k.shape[0] for k,v in states.values()))
            def forward(ids,mode,kv=None):
                active.update(mode=mode,kv=kv,pos=torch.arange(offset,offset+ids.shape[1],device='cuda').unsqueeze(0))
                return model(input_ids=ids,position_ids=active['pos'],use_cache=False).logits.float()
            def generate(q,mode,kv):
                ids=closed_think_inputs(tok,q,'cuda')['input_ids'];n=ids.shape[1]
                for _ in range(24):
                    x=forward(ids,mode,kv)[:,-1].argmax(-1).reshape(1,1);ids=torch.cat((ids,x),1)
                    if x.item()==tok.eos_token_id:break
                token=ids[0,n:].tolist();return {'tokens':token,'text':tok.decode(token,skip_special_tokens=True).strip(),'ended':bool(token and token[-1]==tok.eos_token_id)}
            q='小林最喜欢的饮品是什么？请简洁回答。';ids=closed_think_inputs(tok,q,'cuda')['input_ids']
            base=forward(ids,'off');recomputed=forward(ids,'self')
            check={'max_abs_logits':float((base-recomputed).abs().max()),'same_argmax':bool(torch.equal(base.argmax(-1),recomputed.argmax(-1)))}
            rows=[]
            for c in cases:
                q=f'{c[0]}最喜欢的{c[1]}是什么？请简洁回答。'
                for mode in ('off','self','kernel','oracle'):
                    ans=generate(q,mode,states[c]);rows.append({'case':c,'question':q,'mode':mode,'contains_target':c[2] in ans['text'],**ans})
                save('progress.json',{'check':check,'rows':rows});print(json.dumps({'cases':len(rows)//4,'check':check}),flush=True)
        for h in hooks:h.remove()
        after={k:sha(v) for k,v in paths.items()}
        result={'rows':rows,'self_reconstruction':check,'offset':offset,'input_hashes':hashes,'input_hashes_after':after,
                'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'limitations':['Artificial externally selected raw past KV is not neural compression or durable self-writing.',
                               'Single layer and KV head; fixed templates; target presence is not correctness.',
                               'Only one prompt compares original attention to manual reconstruction.']}
        save('result.json',result);print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':result['seconds']}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
