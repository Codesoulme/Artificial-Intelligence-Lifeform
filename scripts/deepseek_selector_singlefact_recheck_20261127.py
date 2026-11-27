"""Train a tiny question-conditioned neural evidence selector on synthetic dialogues."""
import json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_dialogue_distill_20261127 import REL,statement,question,dialogue

OUT=ROOT/'runs/research/selector-singlefact-recheck-20261127-v1'
CANDIDATE=ROOT/'runs/research/selector-multifact-train-20261127-v1/candidate.pt'
REPORT=ROOT/'runs/research/selector-multifact-train-20261127-v1/result.json'
SEED,STEPS,LR,DIM=112741,0,0.0005,64

class Selector(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.key=nn.Linear(width,DIM,bias=False)
        self.query=nn.Linear(width,DIM,bias=False)
        self.null=nn.Linear(width,1)
    def forward(self,hidden,query):
        x=F.layer_norm(hidden.float(),(hidden.shape[-1],))
        q=F.layer_norm(query.float(),(query.shape[-1],))
        scores=(self.key(x)@self.query(q))/DIM**0.5
        return torch.cat((scores,self.null(q).reshape(1)))

def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    def save(name,data):
        p=OUT/(name+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');p.replace(OUT/name)
    try:
        random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        weight=MODEL/'model.safetensors';before=sha(weight)
        if before!=WEIGHT_SHA:raise ValueError('Unexpected base')
        names=('小陈','小吴','小赵');relations=list(REL)
        if any(len(set(v))!=len(v) for v in REL.values()):raise ValueError('duplicate values')
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        docs={};queries={};samples=[]
        with torch.no_grad():
            for name in names:
                for r in relations:
                    ids=closed_think_inputs(tok,question(name,r),'cuda')['input_ids']
                    queries[(name,r)]=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,-1].detach().cpu()
                vals=range(2) if name!='小赵' else range(2,3)
                for r in relations:
                    for vi in vals:
                        case=(name,r,REL[r][vi]);fact=statement(*case);text=dialogue(case)+'\n'
                        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                        ids=encoded['input_ids'].to('cuda');offs=encoded['offset_mapping'][0].tolist()
                        begin=text.find(fact);vpos=text.find(case[2],begin)
                        if begin<0 or vpos<0 or text.count(fact)!=1:raise ValueError('ambiguous target')
                        target=[i for i,(a,b) in enumerate(offs) if a<vpos+len(case[2]) and b>vpos]
                        if not target:raise ValueError('empty target')
                        hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0].detach().cpu()
                        key=(name,r,vi);docs[key]=(hidden,target)
                        split='train' if name!='小赵' else 'held_name_value'
                        other=relations[(relations.index(r)+(1 if split=='train' else 2))%len(relations)]
                        if other==r or statement(name,other,REL[other][0]) in text:raise ValueError('negative collision')
                        samples.append({'key':key,'query':(name,r),'positive':True,'split':split})
                        samples.append({'key':key,'query':(name,other),'positive':False,'split':split})
                print(json.dumps({'encoded_person':names.index(name)+1,'documents':len(docs)}),flush=True)
        del model;torch.cuda.empty_cache()
        selector=Selector(next(iter(queries.values())).shape[0]).to('cuda')
        report=json.loads(REPORT.read_text(encoding='utf8'))
        if sha(CANDIDATE)!=report['candidate_sha256']: raise ValueError('candidate digest mismatch')
        payload=torch.load(CANDIDATE,map_location='cpu',weights_only=True)
        if payload['base_sha256']!=WEIGHT_SHA: raise ValueError('candidate base mismatch')
        selector.load_state_dict(payload['selector'])
        opt=torch.optim.AdamW(selector.parameters(),lr=LR,weight_decay=0);rng=random.Random(SEED)
        train=[s for s in samples if s['split']=='train'];order=train.copy();trace=[]
        for step in range(STEPS):
            if step%len(order)==0:rng.shuffle(order)
            row=order[step%len(order)];hidden,target=docs[row['key']];q=queries[row['query']]
            scores=selector(hidden.to('cuda'),q.to('cuda'));logp=F.log_softmax(scores,dim=0)
            loss=-logp[target].mean() if row['positive'] else -logp[-1]
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(selector.parameters(),1,error_if_nonfinite=True));opt.step()
            trace.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':norm})
            if step%48==0 or step==STEPS-1:save('training-progress.json',{'trace':trace});print(json.dumps(trace[-1]),flush=True)
        selector.eval();rows=[]
        with torch.no_grad():
            for row in samples:
                hidden,target=docs[row['key']];q=queries[row['query']]
                scores=selector(hidden.to('cuda'),q.to('cuda'));ranking=torch.topk(scores[:-1],min(2,len(hidden))).indices.tolist()
                chose_null=bool((scores[-1]>scores[:-1].max()).item())
                rows.append({'split':row['split'],'case':row['key'],'query_relation':row['query'][1],
                    'positive':row['positive'],'source_tokens':len(hidden),'target_token_count':len(target),
                    'predicted_null':chose_null,'top2_has_any':bool(set(ranking)&set(target)),
                    'top2_has_all':set(target)<=set(ranking),'top2_indexes':ranking,
                    'null_margin':float(scores[-1]-scores[:-1].max())})
        torch.save({'selector':{k:v.detach().cpu() for k,v in selector.state_dict().items()},
                    'base_sha256':WEIGHT_SHA,'seed':SEED,'dim':DIM},OUT/'candidate.pt')
        after=sha(weight)
        save('result.json',{'rows':rows,'trace':trace,'seed':SEED,'steps':STEPS,'lr':LR,'dim':DIM,
            'base_sha_before':before,'base_sha_after':after,'candidate_sha256':sha(OUT/'candidate.pt'),
            'parameter_count':sum(p.numel() for p in selector.parameters()),'seconds':time.perf_counter()-start,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Synthetic single-fact dialogues with same templates; names and values held out.',
                'Ground-truth value token spans only for supervision and scoring, not inference.',
                'No writer integration, free-answer evaluation, actual autonomous memory, or persistence.',
                'Negative examples defined by construction; score not calibrated on natural data.']})
        print(json.dumps({'complete':True,'unchanged':before==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
