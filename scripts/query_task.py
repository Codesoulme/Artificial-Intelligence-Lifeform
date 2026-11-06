"""P22 bounded long-question and evidence-absence environment."""
import random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.wording_task import ROOT, prompt, batch, audit
from scripts.ladder_task import tasks, sample
from scripts.reasoning_task import reversed_events, changed_query

PREFIXES=('请问:{target}', '问题:{target}', '查询:{target}', '想知道:{target}')
SUFFIXES=('经过这些移动之后现在位于什么地方?',
          '在上述记录全部结束以后最后到了哪里?',
          '按照刚才这些先后发生的移动现在在哪里?',
          '从上面的记录来看目前最终处于哪个地方?')
DEV=((1,2),(3,1));TEST=((2,3),(3,2))
TRAIN=tuple((a,b) for a in range(4) for b in range(4) if (a,b) not in DEV+TEST)
OBJECTS=('杯','球','书','笔','包','盒','灯','椅','瓶','盘','伞','帽','鞋','钟','车','船')

def wording(pair):return PREFIXES[pair[0]]+SUFFIXES[pair[1]]

def unknown(row,rng):
    absent=[o for o in OBJECTS if o not in {o for o,p in row['events']}]
    return dict(row,target=rng.choice(absent),answer='未知',indices=[],latest=0)

def evaluation_rows(split,count=96):
    # Pre-existing object-pair split is disclosed; final wordings are new.
    known=sample(tasks(expanded=True)['update'][split],count,322001 if split=='development' else 323001)
    rng=random.Random(322002 if split=='development' else 323002)
    return known,[unknown(row,rng) for row in known]

def check(model,state,known,absent,wordings,details=False):
    result={}
    for template in wordings:
        positive=audit(model,state,known,template)
        negative=audit(model,state,absent,template)
        changed=audit(model,state,[changed_query(r) for r in known],template)
        reverse=audit(model,state,[reversed_events(r) for r in known],template)
        item=dict(known=positive['accuracy'],unknown=negative['accuracy'],
            known_unknown_pair=sum(a['correct'] and b['correct'] for a,b in zip(positive['details'],negative['details']))/len(known),
            changed_query_pair=sum(a['correct'] and b['correct'] for a,b in zip(positive['details'],changed['details']))/len(known),
            reversal_pair=sum(a['correct'] and b['correct'] for a,b in zip(positive['details'],reverse['details']))/len(known),
            false_unknown=sum(d['output']['text']=='未知。' and d['output']['ended_with_eos'] for d in positive['details'])/len(known))
        if details:item['records']=dict(positive=positive,negative=negative,changed=changed,reverse=reverse)
        result[str(template)]=item
    return result

def gate_values(result):
    return [1-v if k=='false_unknown' else v for item in result.values() for k,v in item.items() if k!='records']
