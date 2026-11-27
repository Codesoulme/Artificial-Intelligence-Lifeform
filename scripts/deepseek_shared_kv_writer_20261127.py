"""Shared attentive KV compressor, isolated research; no production writes."""
import json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha

OUT=ROOT/'runs/research/shared-kv-writer-20261127-v1'
SEED,STEPS,LR=112729,64,0.003
REL={'饮品':('豆浆','咖啡','牛奶'),'水果':('苹果','香蕉','西瓜')}
HELD={'饮品':('绿茶',),'水果':('芒果',),'职业':('医生','老师')}
NAMES=('小林','小周');HN='小赵'
NORMAL_TRAIN=('请计算一加二。','请解释什么是雨。')
NORMAL_HELD=('请计算三加四。','请说一句鼓励人的话。')

class Writer(nn.Module):
    def __init__(self,layers):
        super().__init__();self.queries=nn.Parameter(torch.zeros(layers,2,2,128))
    def forward(self,source):
        result=[]
        for layer,(key,val) in enumerate(source):
            k=key[0].float();v=val[0].float()
            score=torch.einsum('hnd,hsd->hsn',F.layer_norm(k,(128,)),self.queries[layer])/128**0.5
            weights=score.softmax(-1)
            result.append((torch.einsum('hsn,hnd->hsd',weights,k).unsqueeze(0),
                           torch.einsum('hsn,hnd->hsd',weights,v).unsqueeze(0)))
        return result

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def save(name,data):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/name)
    try:
        random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        weight=MODEL/'model.safetensors';before=sha(weight)
        if before!=WEIGHT_SHA:raise ValueError('base hash mismatch')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        writer=Writer(model.config.num_hidden_layers).to('cuda')
        def plain(q):return closed_think_inputs(tok,q,'cuda')['input_ids']
        header=plain('')[:,:2]
        with torch.no_grad():
            header_cache=tuple((k.detach(),v.detach()) for k,v in model(input_ids=header,use_cache=True).past_key_values.to_legacy_cache())
        def subj(name,rel):return name+('的职业' if rel=='职业' else '最喜欢的'+rel)
        def fact(case):return subj(case[0],case[1])+'是'+case[2]+'。'
        def question(name,rel):return subj(name,rel)+'是什么？请简洁回答。'
        def alternate(name,rel):return f'{name}最爱喝什么？' if rel=='饮品' else f'{name}最爱吃什么水果？' if rel=='水果' else f'{name}从事什么工作？'
        def suffix(q):
            ids=plain(q)
            if not torch.equal(ids[:,:2],header):raise ValueError('Header mismatch')
            return ids[:,2:]
        states={};all_values={r:REL.get(r,())+HELD.get(r,()) for r in set(REL)|set(HELD)}
        with torch.no_grad():
            for name in NAMES+(HN,):
                for rel,values in all_values.items():
                    for value in values:
                        case=(name,rel,value);ids=tok(fact(case)+'\n',add_special_tokens=False,return_tensors='pt').input_ids.to('cuda')
                        states[case]=tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
        def run(q,source,answer=None):
            seq=suffix(q);ids=seq if answer is None else torch.cat((seq,answer),1)
            if source is None:
                return model(input_ids=torch.cat((header,ids),1),use_cache=False).logits.float(),seq.shape[1]+2
            mem=writer(source);entries=[]
            for i,(hk,hv) in enumerate(header_cache):
                entries.append((torch.cat((hk,mem[i][0].to(hk.dtype)),2),torch.cat((hv,mem[i][1].to(hv.dtype)),2)))
            past=DynamicCache.from_legacy_cache(tuple(entries));p=entries[0][0].shape[2]
            logits=model(input_ids=ids,past_key_values=past,attention_mask=torch.ones(1,p+ids.shape[1],device='cuda',dtype=torch.long),
                         cache_position=torch.arange(p,p+ids.shape[1],device='cuda'),use_cache=False).logits.float()
            return logits,seq.shape[1]
        def loss(q,source,answer):
            target=torch.tensor([tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]],device='cuda')
            logits,n=run(q,source,target);selected=logits[:,n-1:-1]
            return F.cross_entropy(selected.reshape(-1,selected.shape[-1]),target.reshape(-1))
        def gen(q,source):
            tokens=[]
            with torch.no_grad():
                for _ in range(24):
                    ids=torch.tensor([tokens],device='cuda',dtype=torch.long) if tokens else None
                    out,_=run(q,source,ids);t=out[:,-1].argmax(-1).item();tokens.append(t)
                    if t==tok.eos_token_id:break
            return {'text':tok.decode(tokens,skip_special_tokens=True).strip(),'tokens':tokens,'ended':bool(tokens and tokens[-1]==tok.eos_token_id)}
        train=[(n,r,v) for n in NAMES for r,values in REL.items() for v in values]
        cases=[('train',('小林','饮品','豆浆')),('train',('小周','水果','苹果')),
               ('held_name',(HN,'饮品','咖啡')),('held_name',(HN,'水果','香蕉')),
               ('held_value',('小林','饮品','绿茶')),('held_value',('小周','水果','芒果')),
               ('held_relation',(HN,'职业','医生')),('held_relation',('小林','职业','老师'))]
        rows=[]
        def evaluate(phase):
            for split,case in cases:
                n,r,v=case;q=question(n,r);alt=alternate(n,r)
                swap_value=next((x for x in all_values[r] if x!=v),None);swap=(n,r,swap_value)
                wrong=(n,'水果','苹果') if r!='水果' else (n,'饮品','豆浆')
                for mode,query,source in [('correct',q,states[case]),('alternate',alt,states[case]),('swap',q,states[swap]),('wrong_relation',q,states[wrong])]:
                    ans=gen(query,source);rows.append({'phase':phase,'split':split,'case':case,'mode':mode,'question':query,
                        'target_in_text':v in ans['text'],'swapped_value_in_text':swap_value in ans['text'],'wrong_value_in_text':wrong[2] in ans['text'],**ans})
                save('progress.json',{'phase':phase,'rows':rows});print(json.dumps({'phase':phase,'cases':sum(x['phase']==phase for x in rows)//4}),flush=True)
        evaluate('before')
        rng=random.Random(SEED);order=train.copy();opt=torch.optim.AdamW(writer.parameters(),lr=LR,weight_decay=0);trace=[]
        for step in range(STEPS):
            if step%len(order)==0:rng.shuffle(order)
            case=order[step%len(order)];n,r,v=case;wrong_r='水果' if r=='饮品' else '饮品'
            q=question(n,r);wrong_q=question(n,wrong_r);normal=NORMAL_TRAIN[step%len(NORMAL_TRAIN)]
            opt.zero_grad(set_to_none=True)
            ce=loss(q,states[case],fact(case));ce.backward()
            reject=loss(wrong_q,states[case],'我不知道。');reject.backward()
            with torch.no_grad():teacher=run(normal,None)[0][:,-1].detach()
            student=run(normal,states[case])[0][:,-1];kl=F.kl_div(F.log_softmax(student,-1),F.softmax(teacher,-1),reduction='batchmean')
            (0.25*kl).backward();grad=float(torch.nn.utils.clip_grad_norm_(writer.parameters(),1.0,error_if_nonfinite=True));opt.step()
            record={'step':step+1,'case':case,'correct_ce':float(ce.detach()),'wrong_relation_ce':float(reject.detach()),'normal_kl':float(kl.detach()),'gradient_norm':grad}
            if not all(torch.isfinite(torch.tensor(record[k])) for k in ('correct_ce','wrong_relation_ce','normal_kl','gradient_norm')):raise ValueError('nonfinite training')
            trace.append(record)
            if step in (0,15,31,47,63):save('training-progress.json',{'trace':trace});print(json.dumps(record,ensure_ascii=True),flush=True)
        torch.save({'writer':{k:v.detach().cpu() for k,v in writer.state_dict().items()},'seed':SEED,'base_sha256':WEIGHT_SHA},OUT/'candidate.pt')
        evaluate('after')
        for split,case in cases:
            q=question(case[0],case[1]);a=gen(q,None);rows.append({'phase':'control','split':split,'case':case,'mode':'off','question':q,'target_in_text':case[2] in a['text'],**a})
            a=gen(fact(case)+'\n'+q,None);rows.append({'phase':'control','split':split,'case':case,'mode':'fact_prompt','question':q,'target_in_text':case[2] in a['text'],**a})
        for q in NORMAL_HELD:
            for mode,source in [('off',None),('forced_unrelated',states[('小林','饮品','豆浆')])]:
                rows.append({'phase':'control','mode':mode,'question':q,**gen(q,source)})
        after=sha(weight);save('result.json',{'rows':rows,'trace':trace,'seed':SEED,'steps':STEPS,'lr':LR,
            'loss_weights':{'correct':1,'wrong':1,'normal_kl':0.25},'base_sha_before':before,'base_sha_after':after,
            'candidate_sha256':sha(OUT/'candidate.pt'),'candidate_bytes':(OUT/'candidate.pt').stat().st_size,
            'writer_parameter_bytes':sum(p.numel()*p.element_size() for p in writer.parameters()),
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Explicit facts and manually selected states; no autonomous writer boundary or routing.',
                           'Synthetic development examples; held values, names and relation are not a final independent benchmark.',
                           'No formal individual or persistent state was read or written; no memory correction or reload validation.',
                           'Language logits KL only measured at prompt end; exact string matches are screening.']})
        print(json.dumps({'complete':True,'unchanged':before==after,'seconds':time.perf_counter()-started}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
