"""Local noncommercial full-prefix and chunked writer controls on fixed development cases."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter

PILOT=ROOT/'runs/research/perltqa-dialogue-pilot-20261127-v1'
RUN=ROOT/'runs/research/relation-coverage-20261127-v1'
OUT=ROOT/'runs/research/perltqa-compression-controls-20261127-v1'
CHUNK=64

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        paths={'base':MODEL/'model.safetensors','candidate':RUN/'candidate.pt','report':RUN/'result.json',
               'selection':PILOT/'selection.json','pilot':PILOT/'result.json'}
        hashes={k:sha(v) for k,v in paths.items()};parent=json.loads(paths['report'].read_text(encoding='utf8'))
        pilot=json.loads(paths['pilot'].read_text(encoding='utf8'))
        if hashes['base']!=WEIGHT_SHA or hashes['candidate']!=parent['candidate_sha256'] or hashes['candidate']!=pilot['input_hashes']['candidate']:
            raise ValueError('Unexpected checkpoint')
        cases=json.loads(paths['selection'].read_text(encoding='utf8'))['cases']
        if len(cases)!=8:raise ValueError('Unexpected case count')
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        payload=torch.load(paths['candidate'],map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('Wrong base')
        writer=ResidualWriter(model.config.num_hidden_layers).to('cuda').eval().requires_grad_(False);writer.load_state_dict(payload['writer'])
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        with torch.no_grad():
            header=plain('')[:,:2];header_cache=tuple((k.detach(),v.detach()) for k,v in model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache())
            def encode(ids):
                return tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
            def with_header(memory):
                return tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache))
            def gen(suffix,prefix_ids,pairs):
                tokens=[]
                if pairs is not None:p=pairs[0][0].shape[2]
                for _ in range(48):
                    cur=torch.cat((suffix,torch.tensor([tokens],device='cuda')),1) if tokens else suffix
                    if pairs is None:logits=model(input_ids=torch.cat((prefix_ids,cur),1),use_cache=False).logits
                    else:logits=model(input_ids=cur,past_key_values=DynamicCache.from_legacy_cache(pairs),attention_mask=torch.ones(1,p+cur.shape[1],device='cuda',dtype=torch.long),cache_position=torch.arange(p,p+cur.shape[1],device='cuda'),use_cache=False).logits
                    t=logits[:,-1].argmax(-1).item();tokens.append(t)
                    if t==tok.eos_token_id:break
                return {'text':tok.decode(tokens,skip_special_tokens=True).strip(),'tokens':tokens,'ended':tokens[-1]==tok.eos_token_id}
            rows=[]
            for i,c in enumerate(cases):
                dialogue=tok(c['dialogue']+'\n',return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
                ids=plain(c['question']+'请简洁回答。')
                if not torch.equal(ids[:,:2],header):raise ValueError('Header mismatch')
                suffix=ids[:,2:];prefix=torch.cat((header,dialogue),1)
                full=tuple((k.detach(),v.detach()) for k,v in model(input_ids=prefix,use_cache=True).past_key_values.to_legacy_cache())
                whole=with_header(writer(encode(dialogue)))
                chunks=[writer(encode(dialogue[:,j:j+CHUNK])) for j in range(0,dialogue.shape[1],CHUNK)]
                combined=[(torch.cat([x[layer][0] for x in chunks],2),torch.cat([x[layer][1] for x in chunks],2)) for layer in range(len(chunks[0]))]
                chunked=with_header(combined)
                for mode,pair in [('full_prefill',None),('full_cache',full),('two_slot',whole),('chunked',chunked)]:
                    out=gen(suffix,prefix,pair);rows.append({'index':i,'mode':mode,'answer':c['answer'],
                        'target_in_text':c['answer'] in out['text'],'dialogue_tokens':dialogue.shape[1],
                        'memory_positions':dialogue.shape[1] if mode in ('full_prefill','full_cache') else pair[0][0].shape[2]-header.shape[1],**out})
                save('progress.json',{'rows':rows});print(json.dumps({'case':i+1,'rows':len(rows)}),flush=True)
                del full,whole,chunks,combined,chunked
        after={k:sha(v) for k,v in paths.items()}
        save('result.json',{'rows':rows,'chunk_size':CHUNK,'input_hashes':hashes,'input_hashes_after':after,
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Development source and manually selected dialogues; no autonomous or persistent memory.',
                'Chunks encoded with reset positions; concatenation changes position alignment and memory length.',
                'Answer substring is a screening metric, not semantic correctness.',
                'Custom split tokenization differs from ordinary end-to-end chat template tokenization.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-started}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
