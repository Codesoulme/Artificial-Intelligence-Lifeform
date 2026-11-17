"""Isolated supervised first-token memory experiment; not a deployed model."""
import hashlib,json,random,sys,time
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
from spora.research_memory_value_transform import MemoryValueTransform
from spora.research_first_token_pair_loss import pair_loss

OUT=ROOT/'runs/deepseek-kernel-counterfactual-train-20261117'
TRAIN_NAMES=('小林','小周'); HELD_NAME='小赵'
TRAIN_VALUES=('豆浆','咖啡','牛奶','可乐'); HELD_VALUES=('绿茶','果汁')
SEED=110622; STEPS=128; OUTPUT_MAP_ENABLED=False; PAIR_WEIGHT=0.0


def main():
    if OUT.exists():raise FileExistsError(OUT)
    started=time.perf_counter();torch.manual_seed(SEED);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    source=INITIAL/'candidate.pt';previous=json.loads((INITIAL/'result.json').read_text(encoding='utf8'))
    if sha(source)!=previous['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:
        raise ValueError('Source hash mismatch')
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    if checkpoint['base_sha256']!=WEIGHT_SHA or checkpoint['layer']!=5 or checkpoint['head']!=0:
        raise ValueError('Candidate identity mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    kernel=LearnedKernel().to('cuda');kernel.load_state_dict(checkpoint['state'])
    value_transform=MemoryValueTransform(128).to('cuda') if OUTPUT_MAP_ENABLED else None
    attention=model.model.layers[5].self_attn;captured={};active={'kv':None,'positions':None}
    def record(name):return lambda mod,args,out:captured.__setitem__(name,out.detach())
    def inject(mod,args):
        if active['kv'] is None:return None
        hidden=args[0];n=hidden.shape[1];raw=captured['q'].view(1,n,12,128).transpose(1,2)
        cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=raw.dtype),active['positions'])
        rotated,_=apply_rotary_pos_emb(raw,raw,cos,sin)
        query=rotated[0,:6].float().reshape(-1,128);keys,values=active['kv']
        qf=F.softplus(kernel.qmap(F.layer_norm(query,(128,))))+1e-5
        kf=F.softplus(kernel.kmap(F.layer_norm(keys,(128,))))+1e-5
        pred=qf@(kf.T@values)/(qf@kf.sum(0)).unsqueeze(-1)
        if value_transform is not None:pred=value_transform(pred)
        delta=pred.reshape(6,n,128).permute(1,0,2).reshape(1,n,768)
        return (torch.cat((hidden[:,:,:768]+delta.to(hidden.dtype),hidden[:,:,768:]),dim=-1),)+args[1:]
    hooks=[getattr(attention,n+'_proj').register_forward_hook(record(n)) for n in ('q','k','v')]
    hooks.append(attention.o_proj.register_forward_pre_hook(inject))
    def question(name):return f'{name}最喜欢的饮品是什么？请简洁回答。'
    def prompt(name):return closed_think_inputs(tok,question(name),'cuda')['input_ids']
    def forced_ids(name,value):
        prefix=f'{name}最喜欢的饮品是';p=tok.encode(prefix,add_special_tokens=False)
        full=tok.encode(prefix+value,add_special_tokens=False)
        if full[:len(p)]!=p or len(full)<=len(p):raise ValueError('Token boundary invalid')
        return torch.cat((prompt(name),torch.tensor([p],device='cuda')),dim=1),full[len(p)]
    def forward(ids,kv,offset):
        active['kv']=kv;active['positions']=torch.arange(offset,offset+ids.shape[1],device='cuda').unsqueeze(0)
        return model(input_ids=ids,position_ids=active['positions'],use_cache=False).logits[:,-1].float()
    def generate(name,kv,offset):
        ids=prompt(name);start=ids.shape[1]
        with torch.no_grad():
            for _ in range(16):
                token=forward(ids,kv,offset).argmax(-1).reshape(1,1);ids=torch.cat((ids,token),1)
                if token.item()==tok.eos_token_id:break
        result=ids[0,start:].tolist()
        return {'answer':tok.decode(result,skip_special_tokens=True).strip(),'token_ids':result}
    try:
        # Store frozen base activations, never original text in the question input.
        states={};pairs=[(n,v) for n in TRAIN_NAMES+(HELD_NAME,) for v in TRAIN_VALUES+HELD_VALUES]
        with torch.no_grad():
            fact_ids={(n,v):tok(f'{n}最喜欢的饮品是{v}。',return_tensors='pt',add_special_tokens=False).input_ids.to('cuda') for n,v in pairs}
            offset=max(x.shape[1] for x in fact_ids.values())
            for pair,ids in fact_ids.items():
                active['kv']=None;model.model(input_ids=ids,use_cache=False);n=ids.shape[1]
                k=captured['k'].view(1,n,2,128).transpose(1,2);v=captured['v'].view(1,n,2,128).transpose(1,2)
                cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=k.dtype),torch.arange(n,device='cuda').unsqueeze(0))
                _,k=apply_rotary_pos_emb(k,k,cos,sin)
                states[pair]=(k[0,0].float().detach(),v[0,0].float().detach())
        train=[(n,v) for n in TRAIN_NAMES for v in TRAIN_VALUES]
        held=[(HELD_NAME,v) for v in TRAIN_VALUES+HELD_VALUES]+[(TRAIN_NAMES[0],v) for v in HELD_VALUES]
        labels={}
        for pair in pairs:
            ids,label=forced_ids(*pair);labels[pair]=(ids,label)
        for name in TRAIN_NAMES+(HELD_NAME,):
            first=[labels[(name,v)][1] for v in TRAIN_VALUES+HELD_VALUES]
            if len(set(first))!=len(first):raise ValueError('First-token labels not distinct')
        def score(group):
            results=[]
            with torch.no_grad():
                for pair in group:
                    ids,label=labels[pair];logits=forward(ids,states[pair],offset)[0]
                    results.append({'name':pair[0],'value':pair[1],'target':label,
                        'rank':int((logits>logits[label]).sum())+1,'top1':int(logits.argmax())==label})
            return results
        before={'train':score(train),'held':score(held)}
        parameters=list(kernel.parameters())+(list(value_transform.parameters()) if value_transform is not None else [])
        opt=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=0.0)
        order=train.copy();rng=random.Random(SEED);pair_rng=random.Random(SEED+1);trace=[]
        for step in range(STEPS):
            if step%len(order)==0:rng.shuffle(order)
            pair=order[step%len(order)];ids,label=labels[pair]
            opt.zero_grad(set_to_none=True)
            logits=forward(ids,states[pair],offset)
            loss=F.cross_entropy(logits,torch.tensor([label],device='cuda'))
            paired_loss=None
            if PAIR_WEIGHT:
                other=pair_rng.choice([v for v in TRAIN_VALUES if v!=pair[1]])
                other_ids,other_label=labels[(pair[0],other)]
                if not torch.equal(ids,other_ids):raise ValueError('Counterfactual prompts differ')
                other_logits=forward(ids,states[(pair[0],other)],offset)
                contrast=pair_loss(logits,other_logits,label,other_label)
                other_ce=F.cross_entropy(other_logits,torch.tensor([other_label],device='cuda'))
                paired_loss=0.5*other_ce+contrast
                loss=loss+PAIR_WEIGHT*paired_loss
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
            loss.backward()
            map_grad=(float(value_transform.delta.weight.grad.norm()) if value_transform is not None else None)
            grad=float(torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True))
            opt.step()
            if step in (0,15,31,63,127):
                trace.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':grad,'value_map_gradient_norm':map_grad,
                              'paired_loss':float(paired_loss.detach()) if paired_loss is not None else None})
                print(json.dumps(trace[-1]),flush=True)
        after={'train':score(train),'held':score(held)}
        responses=[]
        for pair in held:
            responses.append({'name':pair[0],'value':pair[1],**generate(pair[0],states[pair],offset)})
        no_memory=generate(HELD_NAME,None,offset)
    finally:
        for h in hooks:h.remove()
    if sha(source)!=previous['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:
        raise ValueError('Protected source changed')
    OUT.mkdir()
    torch.save({'state':{k:v.detach().cpu() for k,v in kernel.state_dict().items()},'base_sha256':WEIGHT_SHA,
                'value_transform':({k:v.detach().cpu() for k,v in value_transform.state_dict().items()}
                                   if value_transform is not None else None),
                'parent_sha256':previous['candidate_sha256'],'seed':SEED,'layer':5,'head':0},OUT/'candidate.pt')
    result={'status':'research_only','seed':SEED,'steps':STEPS,'output_map_enabled':OUTPUT_MAP_ENABLED,'pair_weight':PAIR_WEIGHT,'train_names':TRAIN_NAMES,'held_name':HELD_NAME,
            'train_values':TRAIN_VALUES,'held_values':HELD_VALUES,'offset':offset,'before':before,'after':after,
            'held_responses':responses,'held_no_memory':no_memory,'trace':trace,'candidate_sha256':sha(OUT/'candidate.pt'),
            'base_sha256':WEIGHT_SHA,'parent_sha256':previous['candidate_sha256'],
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Artificial single-relation facts, only first answer token directly supervised.',
                           'Known subject and explicitly supplied fact; no autonomous writes or multiple facts per state.',
                           'Question contains no fact text; no past cache, one layer and one KV head only.',
                           'Single run, fixed templates and tiny value/name splits, no normal-chat safety test.',
                           'No independent reload, formal individual changes or durable memory demonstrated.']}
    with (OUT/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'before':before,'after':after,'responses':responses,'seconds':result['seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
