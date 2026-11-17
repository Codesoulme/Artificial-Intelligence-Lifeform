"""Synthetic name-disjoint neural selector; no answer generation or writes."""
import hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from spora.research_identity_selector import IdentitySelector
from scripts.deepseek_kernel_value_span_audit_20261117 import value_token_indices

OUT=ROOT/'runs/deepseek-identity-selector-near-name-probe-20261117'
TRAIN=('小林','小周','小王','小李','小陈','小吴','小郑','小冯','小韩','小朱','小秦','小许',
       '阿宁','阿明','阿华','阿东','阿飞','阿杰','老徐','老马','老丁','老罗','老曹','老刘')
HELD=('小赵','小钱','小孙','小何','阿雅','阿珍','老方','老田','小林子','小王子','阿宁宁','老马力')
TRAIN_Q=('{name}喜欢喝什么？','请问{name}的饮品偏好是什么？')
HELD_Q=('告诉我{name}爱喝哪种饮料。','{name}的饮品喜好是什么？')
RECORD='这条记录属于{name}。'
SEED=110623
USE_ORACLE_SUBJECT_SPAN=False


def main():
    if OUT.exists():raise FileExistsError(OUT)
    if set(TRAIN)&set(HELD) or len(set(TRAIN+HELD))!=len(TRAIN)+len(HELD):raise ValueError('Name leakage')
    started=time.perf_counter();torch.manual_seed(SEED);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    if sha(MODEL/'model.safetensors')!=WEIGHT_SHA:raise ValueError('Base hash mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
          torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    # Identity text is manually supplied separately as the record's identity;
    # no automatic extraction from facts is claimed.
    def feature(text,subject=None):
        encoded=tok(text,return_tensors='pt',return_offsets_mapping=True,add_special_tokens=False)
        ids=encoded.input_ids.to('cuda')
        positions=(value_token_indices(text,subject,encoded.offset_mapping[0].tolist())
                   if USE_ORACLE_SUBJECT_SPAN and subject is not None else None)
        with torch.inference_mode():
            hidden=model.model(input_ids=ids,use_cache=False).last_hidden_state[0].float()
            result=(hidden[positions].mean(0) if positions is not None else hidden[-1]).cpu()
        return result.clone()
    records={};queries={}
    for name in TRAIN+HELD:
        records[name]=feature(RECORD.format(name=name),name)
        templates=TRAIN_Q if name in TRAIN else HELD_Q
        queries[name]=[feature(template.format(name=name),name) for template in templates]
    ordinary=[feature(x) for x in ('今天的天气怎么样？','三加四等于多少？','红灯时行人应当怎么做？')]
    del model;torch.cuda.empty_cache()
    selector=IdentitySelector();optimizer=torch.optim.AdamW(selector.parameters(),lr=0.001,weight_decay=0.001)
    rng=random.Random(SEED)
    trace=[]
    for step in range(2400):
        name=rng.choice(TRAIN);known=(step%2==0)
        names=rng.sample([n for n in TRAIN if n!=name],3)
        if known:names.insert(rng.randrange(4),name)
        else:names.append(rng.choice([n for n in TRAIN if n!=name and n not in names]))
        target=names.index(name) if known else 4
        q=queries[name][rng.randrange(len(TRAIN_Q))].unsqueeze(0)
        rec=torch.stack([records[n] for n in names]).unsqueeze(0)
        logits=selector(q,rec)
        loss=F.cross_entropy(logits,torch.tensor([target]))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        if step in (0,199,799,1599,2399):trace.append({'step':step+1,'loss':float(loss.detach())})
    # Fixed episodes constructed only from held names; known vs unknown queries.
    # These are post-hoc dev cases, not an independent external benchmark.
    rng=random.Random(SEED+71);rows=[];train_rows=[]
    with torch.no_grad():
        for split,nameset,destination in (('train',TRAIN,train_rows),('held',HELD,rows)):
          for name in nameset:
            for template_i in range(2):
                for known in (True,False):
                    others=rng.sample([n for n in nameset if n!=name],3 if known else 4)
                    if known:others.insert(rng.randrange(4),name)
                    target=others.index(name) if known else 4
                    scores=selector(queries[name][template_i].unsqueeze(0),torch.stack([records[n] for n in others]).unsqueeze(0))[0]
                    destination.append({'query_name':name,'query_template':template_i,'known':known,'candidate_names':others,
                                        'target':target,'prediction':int(scores.argmax()),'scores':scores.tolist()})
        ordinary_rows=[]
        for i,q in enumerate(ordinary):
            others=list(HELD[:4]);scores=selector(q.unsqueeze(0),torch.stack([records[n] for n in others]).unsqueeze(0))[0]
            ordinary_rows.append({'case':i,'prediction':int(scores.argmax()),'null_index':4,'scores':scores.tolist()})
        near_rows=[]
        for name,base in (('小林子','小林'),('小王子','小王'),('阿宁宁','阿宁'),('老马力','老马')):
            candidates=[base]+[n for n in TRAIN if n!=base][:3]
            for template_i in range(2):
                scores=selector(queries[name][template_i].unsqueeze(0),torch.stack([records[n] for n in candidates]).unsqueeze(0))[0]
                near_rows.append({'query_name':name,'candidate_names':candidates,'template':template_i,
                                  'prediction':int(scores.argmax()),'null_index':4,'scores':scores.tolist()})
    if sha(MODEL/'model.safetensors')!=WEIGHT_SHA:raise ValueError('Base changed')
    OUT.mkdir()
    torch.save({'state':selector.state_dict(),'base_sha256':WEIGHT_SHA,'seed':SEED,'width':1536,'rank':64},OUT/'candidate.pt')
    def stats(data,group):
        subset=[x for x in data if x['known']==group]
        return {'total':len(subset),'correct':sum(x['target']==x['prediction'] for x in subset),
                'wrong_record':sum(x['prediction']!=4 and x['prediction']!=x['target'] for x in subset),
                'null':sum(x['prediction']==4 for x in subset)}
    result={'seed':SEED,'oracle_subject_span':USE_ORACLE_SUBJECT_SPAN,'base_sha256':WEIGHT_SHA,'candidate_sha256':sha(OUT/'candidate.pt'),
            'train_names':TRAIN,'held_names':HELD,'train_templates':TRAIN_Q,'held_templates':HELD_Q,
            'known':stats(rows,True),'unknown':stats(rows,False),
            'train_known':stats(train_rows,True),'train_unknown':stats(train_rows,False),
            'ordinary':ordinary_rows,'near_name':near_rows,'rows':rows,'train_rows':train_rows,'trace':trace,
            'seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Synthetic isolated names, oracle supplies record subject text.',
                           'Question/record feature pooling depends on oracle_subject_span; no fact state is read.',
                           'No alias, homonym, multi-person context or independently annotated natural dialogue.',
                           'Held names and templates are local split only; base pretraining exposure unknown.',
                           'Null selection is a read refusal, not a generated truthful response.']}
    with (OUT/'result.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'known':result['known'],'unknown':result['unknown'],'ordinary':ordinary_rows,'seconds':result['seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
