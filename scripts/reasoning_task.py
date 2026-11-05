"""Finite external environment and audit oracle, never called by the model."""
import hashlib,itertools,json,random,re,sys
from collections import defaultdict
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.language_bridge import encode,PAD
from spora.short_reasoning import generate_equal_prefixes

OBJECTS=('杯','球','书','笔','包','盒')
PLACES=('桌','柜','箱','架')


def label(events, target):
    indices=[i+1 for i,(obj,place) in enumerate(events) if obj==target]
    if not indices:raise ValueError('Task requires an observed target')
    latest=indices[-1];answer=events[latest-1][1]
    trace='据:'+','.join(map(str,indices))+';后:'+str(latest)+';答:'+answer+'。'
    return {'indices':indices,'latest':latest,'answer':answer,'trace':trace}


def prompt(row,template=0):
    events=''.join(f'{i+1}:{obj}到{place};' for i,(obj,place) in enumerate(row['events']))
    target=row['target']
    question=(f'问:{target}现在哪?',f'问:{target}最后在哪?',f'问题:{target}的位置?')[template]
    return '记录:'+events+question+'答:'


def dataset():
    result={'train':[],'development':[],'test':[]}
    for index,pair in enumerate(itertools.combinations(OBJECTS,2)):
        split='train' if index%5<3 else ('development' if index%5==3 else 'test')
        for repeated,single in (pair,pair[::-1]):
            for locations in itertools.permutations(PLACES,3):
                for positions in itertools.combinations(range(3),2):
                    events=[(repeated if i in positions else single,locations[i]) for i in range(3)]
                    for target in pair:
                        row={'events':events,'target':target,'pair':list(pair)}
                        row.update(label(events,target))
                        result[split].append(row)
    return result


def balanced(rows,count,seed):
    if count%8:raise ValueError('Balanced answer/support count must be divisible by 8')
    groups=defaultdict(list)
    for row in rows:groups[(row['answer'],len(row['indices']))].append(row)
    generator=random.Random(seed);chosen=[]
    for key,values in sorted(groups.items()):chosen.extend(generator.sample(values,count//8))
    generator.shuffle(chosen)
    return chosen


def batch(rows,arm,templates):
    sequences=[];starts=[]
    for row,template in zip(rows,templates):
        prefix=encode(prompt(row,template))[:-1]
        response=row['answer']+'。' if arm=='answer' else row['trace']
        sequences.append(prefix+encode(response)[1:]);starts.append(len(prefix))
    width=max(map(len,sequences))
    if width>256:raise ValueError('Task exceeds context')
    return torch.tensor([seq+[PAD]*(width-len(seq)) for seq in sequences]),torch.tensor(starts)


def changed_query(row):
    other=next(obj for obj in row['pair'] if obj!=row['target'])
    return dict(row,target=other,**label(row['events'],other))


def reversed_events(row):
    events=list(reversed(row['events']))
    return dict(row,events=events,**label(events,row['target']))


def score(row,output,arm):
    valid=output['utf8_valid'] and output['ended_with_eos']
    text=output['text']
    if arm=='answer':
        return {'answer_correct':valid and text==row['answer']+'。','trace_correct':None,
            'evidence_correct':None,'latest_correct':None}
    match=re.fullmatch(r'据:([1-3](?:,[1-3])?);后:([1-3]);答:([桌柜箱架])。',text) if valid else None
    return {'answer_correct':bool(match and match[3]==row['answer']),
        'trace_correct':valid and text==row['trace'],
        'evidence_correct':bool(match and [int(i) for i in match[1].split(',')]==row['indices']),
        'latest_correct':bool(match and int(match[2])==row['latest'])}


def audit(model,state,rows,arm,template=0):
    groups=defaultdict(list);outputs=[None]*len(rows)
    for i,row in enumerate(rows):
        prefix=encode(prompt(row,template))[:-1];groups[len(prefix)].append((i,prefix))
    for items in groups.values():
        for start in range(0,len(items),16):
            subset=items[start:start+16]
            generated=generate_equal_prefixes(model,state,[p for _,p in subset],budget=48)
            for (i,_),output in zip(subset,generated):outputs[i]=output
    details=[dict(prompt=prompt(row,template),expected=row,output=out,**score(row,out,arm)) for row,out in zip(rows,outputs)]
    report={'count':len(rows),'answer_accuracy':sum(r['answer_correct'] for r in details)/len(rows),
        'valid_utf8':sum(r['output']['utf8_valid'] for r in details)/len(rows),
        'ended_with_eos':sum(r['output']['ended_with_eos'] for r in details)/len(rows),'details':details}
    if arm!='answer':
        for key in ('trace_correct','evidence_correct','latest_correct'):report[key]=sum(r[key] for r in details)/len(rows)
    return report
