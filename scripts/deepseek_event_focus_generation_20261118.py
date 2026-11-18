"""Identical-question, focus-only intervention on a frozen integrated candidate.

Finite values and explicit event binding; this does not test reference parsing.
No fact text, value labels or expected answers enter the generation prompt.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from spora.event_focus import observe_focused,bind_event_focus,clear_event_focus,read_focused_event
from spora.growing import save,load,query
from spora.integrated_checkpoint import load_integrated,IntegratedSPORARuntime
from scripts.spora_event_focus_audit_20261118 import fresh,SOURCE,fingerprint

OUT=ROOT/'runs/deepseek-event-focus-generation-20261118'
MODEL=ROOT/'runs/open-base-deepseek-r1-1.5b-ad9f0ae'
BASE_SHA='58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945'
WORDS=('茶','咖啡','牛奶','果汁','豆浆','可乐','水','酸奶')
QUESTIONS=('这条记忆中的人喜欢喝什么？','请根据当前关注的那次记录，说出当时偏爱的饮品。')


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf8'))


def write(path,data):
    with path.open('x',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


@torch.no_grad()
def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_num_threads(2);torch.manual_seed(55781)
    paths=[SOURCE,Path(__file__),ROOT/'scripts/spora_event_focus_audit_20261118.py']
    paths += [ROOT/'spora'/f for f in ('event_focus.py','growing.py','state_codec.py','internal_memory.py',
                                      'integrated_checkpoint.py','integrated_open_base.py','memory_output.py')]
    paths += [MODEL/f for f in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json')]
    hashes={str(p):sha(p) for p in paths}
    if hashes[str(MODEL/'model.safetensors')]!=BASE_SHA:raise ValueError('Base mismatch')
    model,state=fresh()
    decisions=[]
    for key,value in ((2,0),(5,1),(2,2),(5,3)):
        state,decision=observe_focused(model,state,key,value,'direct',1.)
        if not decision['current_write']:raise ValueError('Fixture write rejected')
        decisions.append(decision)
    OUT.mkdir()
    fixtures=[]
    for name,key,event,value in (('a_original',2,0,0),('b_original',5,1,1),('a_corrected',2,2,2),('b_corrected',5,3,3)):
        focused=bind_event_focus(model,state,key,event)
        k=torch.tensor([key]);logits=query(model,focused,k,torch.ones_like(k),torch.full_like(k,event))
        observed=int(logits.argmax(-1)[0])
        if observed!=value:raise ValueError('Fixture neural read does not match observation')
        path=OUT/(name+'.pt');save(path,model,focused)
        fixtures.append({'name':name,'key':key,'event':event,'value':value,'expected_word':WORDS[value],
                         'checkpoint_sha256':sha(path),'state_digest':fingerprint(model,focused)})
    save(OUT/'no_focus.pt',model,clear_event_focus(model,state))
    if any(sha(p)!=hashes[str(p)] for p in paths):raise ValueError('Source changed')
    write(OUT/'plan.json',{'source_sha256':hashes,'fixtures':fixtures,'decisions':decisions,
          'no_focus_sha256':sha(OUT/'no_focus.pt'),'questions':QUESTIONS,
          'conditions':['reader_only','reader_and_finite_output','disabled'],
          'max_new_tokens':24,'do_sample':False,'prefix':'chat template plus explicitly closed think section',
          'scope':'Four focus pointers over identical final historical matrices; finite trained beverages only.',
          'limits':['No learned object reference resolution.','Finite output adapter affects first token only.',
                    'No training or generalization claim.','Explicitly closed think prefix retained from prior diagnostics.']})
    print(json.dumps({'prepared':len(fixtures),'questions':len(QUESTIONS)}),flush=True)


def check():
    plan=read(OUT/'plan.json')
    for name,digest in plan['source_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('Changed source: '+name)
    for fixture in plan['fixtures']:
        if sha(OUT/(fixture['name']+'.pt'))!=fixture['checkpoint_sha256']:raise ValueError('Fixture changed')
    if sha(OUT/'no_focus.pt')!=plan['no_focus_sha256']:raise ValueError('Empty focus changed')
    return plan


def prefix(tokenizer,question):
    text=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=False,add_generation_prompt=True)
    if '<think>' not in text or '</think>' in text:raise ValueError('Unexpected template')
    return text+'</think>\n\n'


def summarize(rows):
    summary={}
    for condition in ('reader_only','reader_and_finite_output'):
        selected=[r for r in rows if r['condition']==condition]
        per_prompt=[]
        for question in QUESTIONS:
            group=[r for r in selected if r['question']==question]
            pairs=[]
            for obj in ('a','b'):
                pair=[r for r in group if r['fixture'].startswith(obj+'_')]
                pairs.append(len(pair)==2 and all(r['exact_word'] for r in pair))
            per_prompt.append({'question':question,'correct_pairs':sum(pairs),'pairs':2,
                               'distinct_outputs':len({r['response'] for r in group})})
        summary[condition]={'exact_words':sum(r['exact_word'] for r in selected),'total':len(selected),
                            'ended':sum(r['ended'] for r in selected),'by_question':per_prompt}
    return summary


@torch.inference_mode()
def generate(reload=False):
    target=OUT/('reload.json' if reload else 'result.json')
    if target.exists():raise FileExistsError(target)
    plan=check()
    reference=read(OUT/'result.json') if reload else None
    if reload and (reference['plan_sha256']!=sha(OUT/'plan.json') or not reference.get('complete')):
        raise ValueError('Reference incomplete or plan changed')
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(2);torch.manual_seed(55781)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
    free,total=torch.cuda.mem_get_info()
    if free<3.5*1024**3:raise RuntimeError('Less than 3.5 GiB free GPU memory; no automatic competing load')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
         torch_dtype=torch.float16,attn_implementation='sdpa',device_map={'':'cuda'},low_cpu_mem_usage=True).eval()
    base.requires_grad_(False)
    print('Base loaded',flush=True)
    decoder,parent_model,parent_state=load_integrated(SOURCE,base,MODEL,individual_device='cuda')
    del parent_model,parent_state
    decoder.requires_grad_(False)
    gc.collect();torch.cuda.empty_cache()
    print('Integrated candidate loaded',flush=True)
    rows=[]
    partial=OUT/('reload-progress.jsonl' if reload else 'progress.jsonl')
    if partial.exists():raise FileExistsError('Partial log exists; inspect prior run before restarting')
    for question in plan['questions']:
        text=prefix(tok,question)
        inputs=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
        for condition in plan['conditions']:
            fixtures=plan['fixtures'][:1] if condition=='disabled' else plan['fixtures']
            for fixture in fixtures:
                organism,state,_=load(OUT/(fixture['name']+'.pt'),'cuda')
                organism.requires_grad_(False)
                if fingerprint(organism,state)!=fixture['state_digest']:raise ValueError('State restore mismatch')
                runtime=IntegratedSPORARuntime(decoder,organism,state,MODEL)
                settings={'max_new_tokens':plan['max_new_tokens'],'do_sample':False,
                          'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id,'use_cache':False}
                before=time.perf_counter()
                if condition=='disabled':ids=decoder.generate(**inputs,**settings)
                else:ids=runtime.generate(**inputs,use_event_focus=True,
                            memory_output=condition=='reader_and_finite_output',**settings)
                torch.cuda.synchronize()
                tail=ids[0,inputs['input_ids'].shape[1]:].tolist()
                response=tok.decode(tail,skip_special_tokens=True).strip()
                normalized=response.rstrip('。.!！ \n')
                row={'question':question,'prompt':text,'prompt_ids':inputs['input_ids'][0].tolist(),
                     'condition':condition,'fixture':fixture['name'] if condition!='disabled' else None,
                     'response':response,'generated_ids':tail,'ended':tok.eos_token_id in tail,
                     'expected':fixture['expected_word'] if condition!='disabled' else None,
                     'exact_word':condition!='disabled' and normalized==fixture['expected_word'],
                     'seconds':time.perf_counter()-before}
                if fingerprint(organism,state)!=fixture['state_digest']:raise ValueError('Generation mutated memory')
                rows.append(row)
                with partial.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'row':len(rows),'condition':condition,'fixture':row['fixture'],
                                  'response':response,'seconds':row['seconds']},ensure_ascii=False),flush=True)
                del runtime,organism,state
    empty_model,empty_state,_=load(OUT/'no_focus.pt','cuda')
    empty_runtime=IntegratedSPORARuntime(decoder,empty_model,empty_state,MODEL)
    try:empty_runtime.generate(**inputs,use_event_focus=True,max_new_tokens=1)
    except ValueError as exc:empty_check={'rejected':True,'reason':str(exc)}
    else:raise ValueError('No-focus route incorrectly generated')
    check()
    result={'rows':rows,'summary':summarize(rows),'complete':True,'empty_focus':empty_check,
            'plan_sha256':sha(OUT/'plan.json'),'seconds':time.perf_counter()-started,
            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
            'initial_free_gpu':free,'training':False}
    if reload:
        comparisons=[(a['question'],a['condition'],a['fixture'],a['prompt_ids'],a['generated_ids'])==
                     (b['question'],b['condition'],b['fixture'],b['prompt_ids'],b['generated_ids'])
                     for a,b in zip(rows,reference['rows'])]
        result['exact_replay']=len(rows)==len(reference['rows']) and all(comparisons)
        result['reference_sha256']=sha(OUT/'result.json')
        if not result['exact_replay']:raise ValueError('Fresh generation differs; progress log retained')
    write(target,result)
    print(json.dumps({'summary':result['summary'],'seconds':result['seconds'],'exact_replay':result.get('exact_replay')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','generate','reload'))
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:generate(reload=args.stage=='reload')
