"""Frozen full-context vs empty-context synthetic entity binding baseline."""
import json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_dialogue_distill_20261127 import REL,statement,question

OUT=ROOT/'runs/research/full-context-binding-20261127-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        path=MODEL/'model.safetensors';before=sha(path)
        if before!=WEIGHT_SHA:raise ValueError('base mismatch')
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        people=('小夏','小何');absent='小唐';rels=list(REL);rows=[]
        def gen(prompt):
            ids=closed_think_inputs(tok,prompt,'cuda')['input_ids'];tokens=[]
            for _ in range(24):
                current=torch.cat((ids,torch.tensor([tokens],device='cuda')),1) if tokens else ids
                t=model(input_ids=current,use_cache=False).logits[:,-1].argmax(-1).item();tokens.append(t)
                if t==tok.eos_token_id:break
            return {'text':tok.decode(tokens,skip_special_tokens=True).strip(),'tokens':tokens,'ended':tokens[-1]==tok.eos_token_id}
        with torch.no_grad():
            for i,r1 in enumerate(rels):
                r2=rels[(i+3)%len(rels)]
                facts=[(people[0],r1,REL[r1][0]),(people[1],r2,REL[r2][2]),
                       (people[1],r1,REL[r1][1]),(people[0],r2,REL[r2][2])]
                text='记录员：以下是大家刚才分享的情况。\n'
                for fact in facts:text+='记录员：'+statement(*fact)+'谢谢你的分享。\n'
                text+='记录员：今天的记录就到这里。'
                for j,person in enumerate((*people,absent)):
                    target=REL[r1][j] if j<2 else None;other=REL[r1][1-j] if j<2 else None
                    for mode,context in [('full',text),('empty','（空）')]:
                        prompt='请只根据给定资料回答问题。如果资料没有答案，请回答“我不知道。”\n资料：\n'+context+'\n问题：'+question(person,r1)
                        out=gen(prompt);rows.append({'relation':r1,'person_role':'absent' if j==2 else 'first' if j==0 else 'second',
                            'mode':mode,'target':target,'other':other,'target_in_text':target in out['text'] if target else None,
                            'other_in_text':other in out['text'] if other else None,'exact_reject':out['text']=='我不知道。',**out})
                save('progress.json',{'rows':rows});print(json.dumps({'relation':i+1,'rows':len(rows)}),flush=True)
        after=sha(path)
        save('result.json',{'rows':rows,'base_sha_before':before,'base_sha_after':after,'seconds':time.perf_counter()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Synthetic development cases previously inspected; not independent evaluation.',
                'Explicit constrained prompt and short greedy generation, not natural chat.',
                'Exact rejection and answer string are screens, not semantic correctness.',
                'Full text baseline uses no compressed or persistent memory.']})
        print(json.dumps({'complete':True,'unchanged':before==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
