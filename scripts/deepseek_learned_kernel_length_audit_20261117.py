"""Held-partition, cross-length attention-vector audit; research-only."""
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel,DATA,OUT as TRAIN
from spora.research_random_feature_attention import read as random_read, write as random_write

OUT=ROOT/'runs/deepseek-learned-kernel-length-audit-20261117.json'


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    started=time.perf_counter()
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    trained=json.loads((TRAIN/'result.json').read_text(encoding='utf8'))
    test_result=json.loads((ROOT/'runs/deepseek-learned-kernel-test-20261117.json').read_text(encoding='utf8'))
    source=DATA/'pretrain-test.jsonl'; candidate=TRAIN/'candidate.pt'
    if (sha(source)!=manifest['files'][source.name] or sha(candidate)!=trained['candidate_sha256']
            or sha(MODEL/'model.safetensors')!=WEIGHT_SHA):raise ValueError('Input digest mismatch')
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    tokens=[]
    with source.open(encoding='utf8') as f:
        for line in f:
            tokens.extend(tokenizer.encode(json.loads(line)['text'],add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
            if len(tokens)>=1088+4*1040:break
    if len(tokens)<1088+4*1040:raise ValueError('Insufficient test tokens')
    # Skip the four 272-token segments used by the previous test.
    segments=[tokens[1088+i*1040:1088+(i+1)*1040] for i in range(4)]
    hashes=[hashlib.sha256(torch.tensor(x,dtype=torch.int64).numpy().tobytes()).hexdigest() for x in segments]
    known=set(trained['segment_hashes']['train']+trained['segment_hashes']['development']+test_result['segment_hashes'])
    if set(hashes)&known:raise ValueError('Identical full segment already used')
    checkpoint=torch.load(candidate,map_location='cpu',weights_only=True)
    if checkpoint['base_sha256']!=WEIGHT_SHA or checkpoint['layer']!=5 or checkpoint['head']!=0:
        raise ValueError('Candidate identity mismatch')
    kernel=LearnedKernel().to('cuda');kernel.load_state_dict(checkpoint['state']);kernel.eval()
    projection=torch.randn((512,128),generator=torch.Generator().manual_seed(110617)).to('cuda')
    torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
          torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False);capture={};attn=model.model.layers[5].self_attn
    hooks=[getattr(attn,n+'_proj').register_forward_hook(lambda module,args,out,n=n:capture.__setitem__(n,out.detach()))
           for n in ('q','k','v')]
    rows=[]
    try:
        with torch.inference_mode():
            for i,seq in enumerate(segments):
                ids=torch.tensor([seq],device='cuda');n=ids.shape[1]
                model.model(input_ids=ids,use_cache=False)
                q=capture['q'].view(1,n,12,128).transpose(1,2)
                k=capture['k'].view(1,n,2,128).transpose(1,2)
                v=capture['v'].view(1,n,2,128).transpose(1,2)
                cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=q.dtype),
                                              torch.arange(n,device='cuda').unsqueeze(0))
                q,k=apply_rotary_pos_emb(q,k,cos,sin)
                query=q[0,:6,1024:].float().reshape(-1,128)
                for length in (128,256,512,1024):
                    keys=k[0,0,1024-length:1024].float(); values=v[0,0,1024-length:1024].float()
                    target=torch.softmax(query@keys.T/(128**0.5),dim=-1)@values
                    learned=kernel(query,keys,values)
                    random=random_read(query,projection,random_write(keys,values,projection))
                    # Sum learned-feature statistics across successive blocks.
                    qf=F.softplus(kernel.qmap(F.layer_norm(query,(128,))))+1e-5
                    a=torch.zeros(128,128,device='cuda');z=torch.zeros(128,device='cuda')
                    for j in range(0,length,128):
                        kf=F.softplus(kernel.kmap(F.layer_norm(keys[j:j+128],(128,))))+1e-5
                        a+=kf.T@values[j:j+128];z+=kf.sum(0)
                    chunk=qf@a/(qf@z).unsqueeze(-1)
                    rows.append({'segment':i,'history_length':length,
                                 'learned_mse':float(F.mse_loss(learned,target)),
                                 'random_mse':float(F.mse_loss(random,target)),
                                 'mean_mse':float(F.mse_loss(values.mean(0).expand_as(target),target)),
                                 'chunk_max_abs':float((chunk-learned).abs().max())})
                print(json.dumps({'evaluated':i+1}),flush=True)
    finally:
        for h in hooks:h.remove()
    if sha(source)!=manifest['files'][source.name] or sha(candidate)!=trained['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:
        raise ValueError('Protected source changed')
    result={'candidate_sha256':trained['candidate_sha256'],'base_sha256':WEIGHT_SHA,
            'source_sha256':manifest['files'][source.name],'segment_hashes':hashes,'rows':rows,
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Four adjacent test-partition segments, not independent samples.',
                           'Full earlier-layer history visible; no text generation.',
                           'One KV head, one candidate seed, one random baseline seed.',
                           'No partial semantic overlap checks or true production persistence.']}
    with OUT.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'seconds':result['seconds'],'peak':result['peak_allocated_bytes'],'rows':len(rows)}),flush=True)

if __name__=='__main__':main()
