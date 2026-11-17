"""Independent-process, held-partition evaluation; no optimizer updates."""
import json
import sys
import time
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_learned_kernel_probe_20261117 import LearnedKernel,corpus_segments,DATA,HISTORY,QUERY,OUT as TRAIN

OUT=ROOT/'runs/deepseek-learned-kernel-test-20261117.json'

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    start=time.perf_counter()
    source=DATA/'pretrain-test.jsonl'
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    trained=json.loads((TRAIN/'result.json').read_text(encoding='utf-8'))
    candidate=TRAIN/'candidate.pt'
    if (sha(source)!=manifest['files'][source.name] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA
            or sha(candidate)!=trained['candidate_sha256']):
        raise ValueError('Source hash mismatch')
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    segments=corpus_segments(source,tokenizer,4)
    hashes=[hashlib.sha256(torch.tensor(seq,dtype=torch.int64).numpy().tobytes()).hexdigest() for seq in segments]
    if set(hashes) & set(trained['segment_hashes']['train']+trained['segment_hashes']['development']):
        raise ValueError('Identical complete segment found in earlier splits')
    ckpt=torch.load(candidate,map_location='cpu',weights_only=True)
    if ckpt['base_sha256']!=WEIGHT_SHA or ckpt['layer']!=5 or ckpt['head']!=0:
        raise ValueError('Candidate identity mismatch')
    kernel=LearnedKernel().to('cuda')
    kernel.load_state_dict(ckpt['state'],strict=True)
    kernel.eval()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
          torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    capture={}
    attn=model.model.layers[5].self_attn
    hooks=[getattr(attn,name+'_proj').register_forward_hook(
        lambda module,args,output,name=name:capture.__setitem__(name,output.detach())) for name in ('q','k','v')]
    rows=[]
    try:
        with torch.inference_mode():
            for index,seq in enumerate(segments):
                ids=torch.tensor([seq],device='cuda'); n=ids.shape[1]
                model.model(input_ids=ids,use_cache=False)
                q=capture['q'].view(1,n,12,128).transpose(1,2)
                k=capture['k'].view(1,n,2,128).transpose(1,2)
                v=capture['v'].view(1,n,2,128).transpose(1,2)
                cos,sin=model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=q.dtype),
                                              torch.arange(n,device='cuda').unsqueeze(0))
                q,k=apply_rotary_pos_emb(q,k,cos,sin)
                query=q[0,:6,HISTORY:].float().reshape(-1,128)
                keys=k[0,0,:HISTORY].float(); values=v[0,0,:HISTORY].float()
                target=torch.softmax(query@keys.T/(128**0.5),dim=-1)@values
                prediction=kernel(query,keys,values)
                rows.append({'segment':index,'mse':float(F.mse_loss(prediction,target)),
                             'mean_baseline_mse':float(F.mse_loss(values.mean(0).expand_as(target),target))})
                print(json.dumps({'evaluated':index+1}),flush=True)
    finally:
        for hook in hooks:hook.remove()
    if sha(source)!=manifest['files'][source.name] or sha(candidate)!=trained['candidate_sha256'] or sha(MODEL/'model.safetensors')!=WEIGHT_SHA:
        raise ValueError('Protected source changed')
    result={'candidate_sha256':trained['candidate_sha256'],'base_sha256':WEIGHT_SHA,
            'source_sha256':manifest['files'][source.name],'segment_hashes':hashes,'rows':rows,
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Four adjacent segments from one test partition; complete segment mismatch only.',
                           'Earlier layers see history, evaluation is vector MSE, no generation.',
                           'Other semantic overlap and base pretraining exposure are unknown.',
                           'One layer, one head, fixed 256-token history, no persistence test.']}
    with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'rows':rows,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
