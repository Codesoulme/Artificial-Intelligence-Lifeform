"""Frozen model attention approximation on local Chinese corpus, research only."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.research_bound_feature_memory import BoundFeatureMemory

CORPUS = ROOT / 'runs/clean-zh-pretrain-v1/pretrain-train.jsonl'
MANIFEST = ROOT / 'runs/clean-zh-pretrain-v1/manifest.json'
OUT = ROOT / 'runs/deepseek-long-kv-feature-probe-20261117.json'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    model_path = MODEL / 'model.safetensors'
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    corpus_sha = sha(CORPUS)
    if sha(model_path) != WEIGHT_SHA or corpus_sha != manifest['files'][CORPUS.name]:
        raise ValueError('Input hash mismatch')
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    # Concatenate first rows with explicit EOS boundaries, then take two
    # adjacent fixed segments. The original corpus text is not exported.
    tokens = []
    with CORPUS.open(encoding='utf-8') as stream:
        for line in stream:
            tokens.extend(tok.encode(json.loads(line)['text'], add_special_tokens=False))
            tokens.append(tok.eos_token_id)
            if len(tokens) >= 2 * 1040:
                break
    if len(tokens) < 2080:
        raise ValueError('Insufficient corpus tokens')
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    captured = {}
    attn = model.model.layers[5].self_attn
    hooks = [getattr(attn, name + '_proj').register_forward_hook(
             lambda module, args, output, name=name: captured.__setitem__(name, output.detach()))
             for name in ('q', 'k', 'v')]
    rows = []
    try:
        for segment in range(2):
            ids = torch.tensor([tokens[segment*1040:(segment+1)*1040]], dtype=torch.long, device='cuda')
            with torch.inference_mode():
                model.model(input_ids=ids, use_cache=False)
                n = ids.shape[1]
                q = captured['q'].view(1, n, 12, 128).transpose(1, 2)
                k = captured['k'].view(1, n, 2, 128).transpose(1, 2)
                v = captured['v'].view(1, n, 2, 128).transpose(1, 2)
                cos, sin = model.model.rotary_emb(torch.zeros(1,n,128,device='cuda',dtype=q.dtype),
                                                  torch.arange(n,device='cuda').unsqueeze(0))
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                for head in range(2):
                    queries = q[0,head*6:(head+1)*6,1024:].float().reshape(-1,128)
                    for length in (128, 512, 1024):
                        keys = k[0,head,1024-length:1024].float()
                        vals = v[0,head,1024-length:1024].float()
                        true = torch.softmax(queries @ keys.T / (128**0.5), dim=-1) @ vals
                        baseline = vals.mean(0).expand_as(true)
                        base_mse = float(F.mse_loss(baseline, true))
                        for seed in (110617, 110618, 110619):
                            p = torch.randn((512,128), generator=torch.Generator().manual_seed(seed)).to('cuda')
                            bound = BoundFeatureMemory(p, base_sha256=WEIGHT_SHA, layer=5, kv_head=head)
                            state = None
                            for i in range(0,length,128):
                                block = bound.write(keys[i:i+128], vals[i:i+128])
                                state = block if state is None else bound.merge(state, block)
                            prediction = bound.read(queries, state)
                            whole = bound.read(queries, bound.write(keys, vals))
                            rows.append({'segment': segment, 'head': head, 'history_length': length,
                                         'seed': seed, 'mse': float(F.mse_loss(prediction,true)),
                                         'baseline_mse': base_mse,
                                         'cosine': float(F.cosine_similarity(prediction,true,dim=-1).mean()),
                                         'chunk_vs_whole_max_abs': float((prediction-whole).abs().max())})
                print(json.dumps({'completed_segment':segment+1}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if sha(model_path) != WEIGHT_SHA or sha(CORPUS) != corpus_sha:
        raise ValueError('Protected input changed')
    result = {'base_sha256':WEIGHT_SHA,'corpus_sha256':corpus_sha,
              'token_ids_sha256':hashlib.sha256(torch.tensor(tokens[:2080],dtype=torch.int64).numpy().tobytes()).hexdigest(),
              'rows':rows, 'seconds':time.perf_counter()-started,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
              'limitations':['Two contiguous corpus segments with possible document boundaries, not independent samples.',
                             'Full history visible to earlier layers; no answer generation.',
                             'One layer; no held-out subjects or natural memory writes.',
                             'Direct full attention is required for evaluation; timing not a deployment cost.']}
    with OUT.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'seconds':result['seconds'],'peak':result['peak_allocated_bytes'],'rows':len(rows)}))


if __name__ == '__main__':
    main()
