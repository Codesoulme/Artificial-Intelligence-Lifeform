"""Fixed-question, swapped-state comparison for an isolated trained gate."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel
from scripts.deepseek_kernel_relation_gate_20261127 import Gate, SOURCE
from spora.research_memory_value_transform import MemoryValueTransform

RUN = ROOT / 'runs/research/relation-gate-20261127-v1'
OUT = RUN / 'counterfactual.json'

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    t = time.perf_counter(); torch.set_num_threads(2); torch.cuda.reset_peak_memory_stats()
    paths = {'base': MODEL/'model.safetensors', 'kernel': SOURCE/'candidate.pt', 'gate': RUN/'gate.pt', 'result': RUN/'result.json'}
    hashes = {k: sha(p) for k,p in paths.items()}
    previous = json.loads(paths['result'].read_text(encoding='utf8'))
    if hashes['base'] != WEIGHT_SHA or hashes['kernel'] != previous['input_hashes']['candidate'] or hashes['gate'] != previous['gate_sha256']:
        raise ValueError('Source mismatch')
    payload = torch.load(paths['kernel'], map_location='cpu', weights_only=True)
    g = torch.load(paths['gate'], map_location='cpu', weights_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    kernel = LearnedKernel().to('cuda').eval(); kernel.load_state_dict(payload['state']); kernel.requires_grad_(False)
    trans = MemoryValueTransform(128).to('cuda').eval(); trans.load_state_dict(payload['value_transform']); trans.requires_grad_(False)
    gate = Gate().to('cuda').eval(); gate.load_state_dict(g['gate']); gate.requires_grad_(False)
    attn = model.model.layers[5].self_attn; captured = {}; active = {'state': None, 'trained': False, 'pos': None}
    def record(name): return lambda mod,args,out: captured.__setitem__(name,out.detach())
    def inject(mod,args):
        if active['state'] is None: return None
        h=args[0]; n=h.shape[1]; raw=captured['q'].view(1,n,12,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=raw.dtype),active['pos'])
        q,_=apply_rotary_pos_emb(raw,raw,cos,sin); q=q[0,:6].float().reshape(-1,128)
        k,v=active['state']; qf=F.softplus(kernel.qmap(F.layer_norm(q,(128,))))+1e-5
        kf=F.softplus(kernel.kmap(F.layer_norm(k,(128,))))+1e-5
        delta=trans(qf@(kf.T@v)/(qf@kf.sum(0)).unsqueeze(-1))
        if active['trained']: delta=delta*gate(q,k)
        delta=delta.reshape(6,n,128).permute(1,0,2).reshape(1,n,768)
        return (torch.cat((h[:,:,:768]+delta.to(h.dtype),h[:,:,768:]),-1),)+args[1:]
    hooks=[getattr(attn,n+'_proj').register_forward_hook(record(n)) for n in ('q','k','v')]
    hooks.append(attn.o_proj.register_forward_pre_hook(inject))
    def make_state(fact):
        ids=tok(fact,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda'); active['state']=None
        model.model(input_ids=ids,use_cache=False); n=ids.shape[1]
        k=captured['k'].view(1,n,2,128).transpose(1,2); v=captured['v'].view(1,n,2,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),torch.arange(n,device='cuda').unsqueeze(0))
        _,k=apply_rotary_pos_emb(k,k,cos,sin); return k[0,0].float(),v[0,0].float()
    def generate(q,state,trained):
        ids=closed_think_inputs(tok,q,'cuda')['input_ids']; start=ids.shape[1]
        for _ in range(24):
            active.update(state=state,trained=trained,pos=torch.arange(g['offset'],g['offset']+ids.shape[1],device='cuda').unsqueeze(0))
            next_id=model(input_ids=ids,position_ids=active['pos'],use_cache=False).logits[:,-1].argmax(-1).reshape(1,1)
            ids=torch.cat((ids,next_id),1)
            if next_id.item()==tok.eos_token_id: break
        output=ids[0,start:].tolist(); return {'text':tok.decode(output,skip_special_tokens=True).strip(), 'tokens':output, 'ended':bool(output and output[-1]==tok.eos_token_id)}
    rows=[]
    with torch.no_grad():
        for name,rel,values in [('小林','饮品',('豆浆','咖啡')),('小赵','饮品',('豆浆','咖啡')),('小林','水果',('苹果','香蕉')),('小赵','水果',('苹果','香蕉'))]:
            q=f'{name}最喜欢的{rel}是什么？请简洁回答。'
            states=[make_state(f'{name}最喜欢的{rel}是{v}。') for v in values]
            for trained in (False,True):
                for value,state in zip(values,states):
                    answer=generate(q,state,trained)
                    rows.append({'name':name,'relation':rel,'value':value,'trained_gate':trained,'question':q,'target_in_text':value in answer['text'],**answer})
                    (RUN/'counterfactual.progress.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
            print(json.dumps({'name':name,'relation':rel,'rows':len(rows)},ensure_ascii=True),flush=True)
    for hook in hooks: hook.remove()
    final={k:sha(p) for k,p in paths.items()}
    result={'rows':rows,'input_hashes':hashes,'input_hashes_after':final,'seconds':time.perf_counter()-t,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'limitations':['Synthetic facts and manually selected states; no independent production memory.','Exact string match is screening, not semantic evaluation.']}
    with OUT.open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'complete':True,'source_unchanged':hashes==final,'seconds':result['seconds']}),flush=True)

if __name__=='__main__': main()
