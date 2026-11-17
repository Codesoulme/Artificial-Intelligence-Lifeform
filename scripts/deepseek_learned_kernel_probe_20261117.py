"""Offline supervised kernel probe; frozen DeepSeek, no individual writes."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

DATA = ROOT / 'runs/clean-zh-pretrain-v1'
OUT = ROOT / 'runs/deepseek-learned-kernel-probe-20261117'
SEED = 110621
HISTORY = 256
QUERY = 16


class LearnedKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qmap = nn.Linear(128, 128, bias=True)
        self.kmap = nn.Linear(128, 128, bias=True)

    def forward(self, q, k, v):
        qf = F.softplus(self.qmap(F.layer_norm(q.float(), (128,)))) + 1e-5
        kf = F.softplus(self.kmap(F.layer_norm(k.float(), (128,)))) + 1e-5
        numerator = qf @ (kf.T @ v.float())
        denominator = qf @ kf.sum(0)
        return numerator / denominator.unsqueeze(-1)


def corpus_segments(path, tokenizer, count):
    tokens = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            tokens.extend(tokenizer.encode(json.loads(line)['text'], add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
            if len(tokens) >= count * (HISTORY + QUERY):
                break
    if len(tokens) < count * (HISTORY + QUERY):
        raise ValueError('Insufficient corpus tokens')
    return [tokens[i*(HISTORY+QUERY):(i+1)*(HISTORY+QUERY)] for i in range(count)]


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    paths = {'train':DATA/'pretrain-train.jsonl', 'development':DATA/'pretrain-development.jsonl'}
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    source = {k:sha(v) for k,v in paths.items()}
    if (sha(MODEL/'model.safetensors') != WEIGHT_SHA
            or any(source[k] != manifest['files'][p.name] for k,p in paths.items())):
        raise ValueError('Source hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    segments = {k:corpus_segments(p,tokenizer,4 if k=='train' else 2) for k,p in paths.items()}
    if set(map(tuple,segments['train'])) & set(map(tuple,segments['development'])):
        raise ValueError('Identical full segment across splits')
    model = AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
              torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    capture = {}
    attention = model.model.layers[5].self_attn
    hooks = [getattr(attention, name+'_proj').register_forward_hook(
             lambda module,args,output,name=name: capture.__setitem__(name,output.detach()))
             for name in ('q','k','v')]
    examples = {'train':[], 'development':[]}
    try:
        with torch.inference_mode():
            for split, seqs in segments.items():
                for seq in seqs:
                    ids=torch.tensor([seq],device='cuda')
                    model.model(input_ids=ids,use_cache=False)
                    n=ids.shape[1]
                    q=capture['q'].view(1,n,12,128).transpose(1,2)
                    k=capture['k'].view(1,n,2,128).transpose(1,2)
                    v=capture['v'].view(1,n,2,128).transpose(1,2)
                    cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=q.dtype),
                                                   torch.arange(n,device='cuda').unsqueeze(0))
                    q,k=apply_rotary_pos_emb(q,k,cos,sin)
                    # One KV head and its six query heads; fixed historic boundary.
                    query=q[0,:6,HISTORY:].float().reshape(-1,128).cpu()
                    keys=k[0,0,:HISTORY].float().cpu()
                    vals=v[0,0,:HISTORY].float().cpu()
                    target=(torch.softmax(query @ keys.T / (128**0.5),dim=-1) @ vals).detach()
                    examples[split].append((query,keys,vals,target))
                    print(json.dumps({'extracted_split':split,'count':len(examples[split])}),flush=True)
    finally:
        for hook in hooks: hook.remove()
    del capture, model
    torch.cuda.empty_cache()
    kernel=LearnedKernel().to('cuda')
    samples={split:[tuple(t.to('cuda') for t in sample) for sample in group] for split,group in examples.items()}
    def evaluate(split):
        rows=[]
        with torch.no_grad():
            for q,k,v,target in samples[split]:
                pred=kernel(q,k,v)
                rows.append({'mse':float(F.mse_loss(pred,target)),
                             'mean_baseline_mse':float(F.mse_loss(v.mean(0).expand_as(target),target))})
        return rows
    before={s:evaluate(s) for s in samples}
    opt=torch.optim.AdamW(kernel.parameters(),lr=0.001,weight_decay=0.0001)
    trace=[]
    for step in range(512):
        q,k,v,target=samples['train'][step%len(samples['train'])]
        pred=kernel(q,k,v)
        loss=F.mse_loss(pred,target)/(target.square().mean().detach()+1e-8)
        if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(kernel.parameters(),1.0,error_if_nonfinite=True)
        opt.step()
        if step in (0,63,127,255,511):
            trace.append({'step':step+1,'normalized_mse':float(loss.detach())})
    after={s:evaluate(s) for s in samples}
    if sha(MODEL/'model.safetensors')!=WEIGHT_SHA or any(sha(p)!=source[k] for k,p in paths.items()):
        raise ValueError('Protected input changed')
    OUT.mkdir()
    torch.save({'state':{k:v.cpu() for k,v in kernel.state_dict().items()},'seed':SEED,
                'base_sha256':WEIGHT_SHA,'layer':5,'head':0},OUT/'candidate.pt')
    result={'seed':SEED,'train_segments':4,'development_segments':2,'history':HISTORY,'query_positions':QUERY,
            'base_sha256':WEIGHT_SHA,'source_sha256':source,
            'segment_hashes':{split:[hashlib.sha256(torch.tensor(seq,dtype=torch.int64).numpy().tobytes()).hexdigest()
                                    for seq in seqs] for split,seqs in segments.items()},
            'before':before,'after':after,'trace':trace,'candidate_sha256':sha(OUT/'candidate.pt'),
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Four training and two development segments, adjacent within split.',
                           'Corpus split does not prove full document/content disjointness or base pretraining novelty.',
                           'Full history visible to earlier model layers; attention-output teacher diagnostic only.',
                           'No generated answers, identity routing, persistence, or production deployment.']}
    with (OUT/'result.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'before':before,'after':after,'seconds':result['seconds']}),flush=True)


if __name__=='__main__': main()
