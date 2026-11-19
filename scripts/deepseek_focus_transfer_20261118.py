"""Broader fixed, untrained focus intervention with value permutations and prose prompts."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from scripts.deepseek_event_focus_generation_20261118 import MODEL,BASE_SHA,WORDS,sha,read,write,prefix
from scripts.spora_event_focus_audit_20261118 import SOURCE,fresh,fingerprint
from spora.event_focus import observe_focused,bind_event_focus,read_focused_event
from spora.growing import save,load,query
from spora.state_codec import flatten
from spora.integrated_checkpoint import load_integrated,IntegratedSPORARuntime

OUT=ROOT/'runs/deepseek-focus-transfer-20261118'
KEYS=(21,27,39,46)
QUESTIONS=(
    {'name':'previous_word','text':'这条记忆中的人喜欢喝什么？','format':'word'},
    {'name':'new_word','text':'当前关注的这次经历里，记录的饮品是哪一种？','format':'word'},
    {'name':'sentence','text':'请用一个完整的中文句子说明：当前关注的那次记录中，这个人喜欢喝什么？','format':'sentence'})


@torch.no_grad()
def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    torch.set_num_threads(2);torch.manual_seed(55782)
    paths=[SOURCE,Path(__file__),ROOT/'scripts/deepseek_event_focus_generation_20261118.py',
           ROOT/'scripts/spora_event_focus_audit_20261118.py']
    paths += [ROOT/'spora'/f for f in ('event_focus.py','growing.py','state_codec.py','internal_memory.py',
                                      'integrated_checkpoint.py','integrated_open_base.py','memory_output.py')]
    paths += [MODEL/f for f in ('model.safetensors','config.json','tokenizer.json','tokenizer_config.json')]
    hashes={str(p):sha(p) for p in paths}
    if hashes[str(MODEL/'model.safetensors')]!=BASE_SHA:raise ValueError('Base changed')
    OUT.mkdir();fixtures=[];decisions=[];inspections=[]
    for world in range(2):
        model,state=fresh()
        schedule=[]
        for version in range(2):
            for i,key in enumerate(KEYS):
                value=(i+version*4+world*3)%len(WORDS)
                event=state.cursor
                state,decision=observe_focused(model,state,key,value,'direct',1.)
                decisions.append({'world':world,'key':key,'value':value,**decision})
                if not decision['current_write']:raise ValueError('Fixture observation rejected')
                schedule.append((key,event,value,version))
        reference_schema,reference_tensors=flatten(model,state)
        for key,event,value,version in schedule:
            name=f'w{world}-k{key}-v{version}'
            focused=bind_event_focus(model,state,key,event)
            schema,tensors=flatten(model,focused)
            differences=[n for n,t in tensors.items() if not torch.equal(t,reference_tensors[n])]
            if schema!=reference_schema or set(differences)-{'internal/event_focus'}:
                raise ValueError('Focus intervention changed other state')
            k=torch.tensor([key]);logits=query(model,focused,k,torch.ones_like(k),torch.full_like(k,event))
            observed=int(logits.argmax(-1)[0])
            if observed!=value:raise ValueError('Neural read failure; do not silently exclude fixture')
            path=OUT/(name+'.pt');save(path,model,focused)
            fixtures.append({'name':name,'world':world,'key':key,'event':event,'version':version,
                'value':value,'expected_word':WORDS[value],'checkpoint_sha256':sha(path),'state_digest':fingerprint(model,focused)})
            inspections.append({'fixture':name,'observed':observed,'logits':logits[0].tolist(),'changed_tensors':differences})
    if any(sha(p)!=hashes[str(p)] for p in paths):raise ValueError('Sources changed')
    write(OUT/'plan.json',{'source_sha256':hashes,'fixtures':fixtures,'decisions':decisions,'inspections':inspections,
        'questions':QUESTIONS,'max_new_tokens':48,'do_sample':False,'memory_output':False,
        'conditions':['reader_only','disabled'],'prior_experiment':'739db1f',
        'scope':'New keys versus preceding focus experiment, all eight already-trained words, two value permutations.',
        'limits':['Not unseen-pretraining or unseen-value evidence.','Explicit finite key/value observations.',
                  'Learned reference parsing and open-text memory not evaluated.',
                  'Word exactness does not assess sentence meaning; sentence outputs require separate review.']})
    print(json.dumps({'fixtures':len(fixtures),'questions':len(QUESTIONS),'neural_reads_correct':len(inspections)}),flush=True)


def check():
    plan=read(OUT/'plan.json')
    for name,digest in plan['source_sha256'].items():
        if sha(Path(name))!=digest:raise ValueError('Source changed: '+name)
    for f in plan['fixtures']:
        if sha(OUT/(f['name']+'.pt'))!=f['checkpoint_sha256']:raise ValueError('State changed')
    return plan


def lexical(response,target):
    stripped=response.strip().rstrip('。.!！ \n')
    return {'exact_word':stripped==target,
            'target_mentioned':target in response,
            'other_words_mentioned':[w for w in WORDS if w!=target and w in response]}


def summarize(rows,plan):
    result={}
    for question in plan['questions']:
        group=[r for r in rows if r['question_name']==question['name'] and r['condition']=='reader_only']
        pairs=[]
        for world in range(2):
            for key in KEYS:
                pair=[r for r in group if r['world']==world and r['key']==key]
                pairs.append(len(pair)==2 and all(r['lexical']['exact_word'] for r in pair))
        result[question['name']]={'total':len(group),'exact_word':sum(r['lexical']['exact_word'] for r in group),
            'target_mentioned':sum(r['lexical']['target_mentioned'] for r in group),
            'only_target_mentioned':sum(r['lexical']['target_mentioned'] and not r['lexical']['other_words_mentioned'] for r in group),
            'ended':sum(r['ended'] for r in group),'exact_word_pairs':sum(pairs),'pairs':len(pairs)}
    return result


@torch.inference_mode()
def generate(reload=False):
    target=OUT/('reload.json' if reload else 'result.json')
    progress=OUT/('reload-progress.jsonl' if reload else 'progress.jsonl')
    if target.exists() or progress.exists():raise FileExistsError('Existing run; inspect before restart')
    plan=check();reference=read(OUT/'result.json') if reload else None
    if reference and reference['plan_sha256']!=sha(OUT/'plan.json'):raise ValueError('Plan changed')
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(2);torch.manual_seed(55782)
    free,total=torch.cuda.mem_get_info()
    if free<3.5*1024**3:raise RuntimeError('Insufficient free GPU memory for bounded run')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
         torch_dtype=torch.float16,attn_implementation='sdpa',device_map={'':'cuda'},low_cpu_mem_usage=True).eval()
    base.requires_grad_(False)
    print('Base loaded',flush=True)
    decoder,parent,old=load_integrated(SOURCE,base,MODEL,individual_device='cuda')
    del parent,old
    decoder.requires_grad_(False);gc.collect();torch.cuda.empty_cache()
    print('Integrated candidate loaded',flush=True)
    rows=[]
    for question in plan['questions']:
        text=prefix(tokenizer,question['text'])
        inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False).to('cuda')
        for condition in plan['conditions']:
            fixtures=plan['fixtures'] if condition=='reader_only' else plan['fixtures'][:1]
            for f in fixtures:
                model,state,_=load(OUT/(f['name']+'.pt'),'cuda');model.requires_grad_(False)
                if fingerprint(model,state)!=f['state_digest']:raise ValueError('Restore mismatch')
                runtime=IntegratedSPORARuntime(decoder,model,state,MODEL)
                settings={'max_new_tokens':plan['max_new_tokens'],'do_sample':False,'use_cache':False,
                          'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id}
                before=time.perf_counter()
                if condition=='reader_only':ids=runtime.generate(**inputs,use_event_focus=True,memory_output=False,**settings)
                else:ids=decoder.generate(**inputs,**settings)
                torch.cuda.synchronize()
                tail=ids[0,inputs['input_ids'].shape[1]:].tolist()
                response=tokenizer.decode(tail,skip_special_tokens=True).strip()
                row={'question_name':question['name'],'question':question['text'],'format':question['format'],
                     'prompt':text,'prompt_ids':inputs['input_ids'][0].tolist(),'condition':condition,
                     'fixture':f['name'] if condition=='reader_only' else None,'world':f['world'],
                     'key':f['key'],'version':f['version'],'response':response,'generated_ids':tail,
                     'ended':tokenizer.eos_token_id in tail,'seconds':time.perf_counter()-before,
                     'expected':f['expected_word'] if condition=='reader_only' else None,
                     'lexical':lexical(response,f['expected_word']) if condition=='reader_only' else None}
                if fingerprint(model,state)!=f['state_digest']:raise ValueError('Inference changed state')
                rows.append(row)
                with progress.open('a',encoding='utf8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({'row':len(rows),'question':question['name'],'fixture':row['fixture'],
                                  'response':response},ensure_ascii=False),flush=True)
                del runtime,model,state
    check()
    result={'rows':rows,'summary':summarize(rows,plan),'plan_sha256':sha(OUT/'plan.json'),
            'seconds':time.perf_counter()-started,'peak_allocated':torch.cuda.max_memory_allocated(),
            'peak_reserved':torch.cuda.max_memory_reserved(),'complete':True,'training':False}
    if reload:
        fields=('question_name','condition','fixture','prompt_ids','generated_ids')
        result['exact_replay']=len(rows)==len(reference['rows']) and all(
            all(a[k]==b[k] for k in fields) for a,b in zip(rows,reference['rows']))
        if not result['exact_replay']:raise ValueError('Reload differs; raw log retained')
        result['reference_sha256']=sha(OUT/'result.json')
    write(target,result)
    print(json.dumps({'summary':result['summary'],'seconds':result['seconds'],'exact_replay':result.get('exact_replay')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','generate','reload'))
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:generate(reload=args.stage=='reload')
