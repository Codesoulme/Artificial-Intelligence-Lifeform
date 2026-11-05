"""Finite movement curriculum; oracle is restricted to labels and evaluation."""
import itertools,random,sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.reasoning_task import ROOT,OBJECTS,PLACES,label,dataset,batch,audit,changed_query,reversed_events
from scripts.reasoning_task import prompt
from spora.language_bridge import encode


def evidence_positions(rows):
    """Label the last UTF-8 byte of the relevant location; training only."""
    positions=[]
    for row in rows:
        text='记录:'
        for i,(obj,place) in enumerate(row['events']):
            text+=f'{i+1}:{obj}到{place}'
            if i+1==row['latest']:
                positions.append(len(text.encode('utf-8')));break
            text+=';'
    return positions


def matching_positions(rows):
    """Training labels: final UTF-8 byte of EVERY matching record location."""
    result=[]
    for row in rows:
        text='记录:';positions=[]
        for i,(obj,place) in enumerate(row['events']):
            text+=f'{i+1}:{obj}到{place}'
            if obj==row['target']:positions.append(len(text.encode('utf-8')))
            text+=';'
        result.append(positions)
    return result


@torch.no_grad()
def evidence_audit(model,state,rows):
    if not getattr(model,'evidence_reader',False):return None
    groups=defaultdict(list);details=[]
    for i,row in enumerate(rows):
        prefix=encode(prompt(row))[:-1];groups[len(prefix)].append((i,prefix,row))
    device=next(model.parameters()).device
    for items in groups.values():
        for start in range(0,len(items),16):
            part=items[start:start+16];tokens=torch.tensor([p for _,p,_ in part],device=device)
            model(tokens,state);chosen=model.evidence_logits[:,-1].argmax(-1).tolist()
            expected=evidence_positions([r for _,_,r in part])
            if getattr(model,'copy_reader',False):expected=[x-2 for x in expected]
            details.extend({'index':i,'predicted_position':actual,'expected_position':target,'correct':actual==target}
                for (i,_,_),actual,target in zip(part,chosen,expected))
    return {'count':len(rows),'accuracy':sum(d['correct'] for d in details)/len(rows),'details':sorted(details,key=lambda x:x['index'])}


def tasks(expanded=False):
    stages={'single':{'train':[],'development':[],'test':[]},'select':{'train':[],'development':[],'test':[]},'update':dataset()}
    objects=OBJECTS+('灯','椅','瓶','盘','伞','帽','鞋','钟','车','船') if expanded else OBJECTS
    for obj,place in itertools.product(objects,PLACES):
        events=[(obj,place)];row=dict(events=events,target=obj,pair=[obj],**label(events,obj))
        # Single-record stage is a complete vocabulary fit check, no held-out claim.
        for split in stages['single']:stages['single'][split].append(row)
    original_pairs=list(itertools.combinations(OBJECTS,2))
    for pair in itertools.combinations(objects,2):
        if pair in original_pairs:
            index=original_pairs.index(pair)
            split='train' if index%5<3 else ('development' if index%5==3 else 'test')
        else:split='train'
        for ordered_objects in (pair,pair[::-1]):
            for locations in itertools.permutations(PLACES,2):
                events=list(zip(ordered_objects,locations))
                for target in pair:stages['select'][split].append(dict(events=events,target=target,pair=list(pair),**label(events,target)))
        if expanded and pair not in original_pairs:
            for repeated,single in (pair,pair[::-1]):
                for locations in itertools.permutations(PLACES,3):
                    for positions in itertools.combinations(range(3),2):
                        events=[(repeated if i in positions else single,locations[i]) for i in range(3)]
                        for target in pair:stages['update']['train'].append(dict(events=events,target=target,pair=list(pair),**label(events,target)))
    return stages


def sample(rows,count,seed):
    groups=defaultdict(list)
    for row in rows:groups[(row['answer'],len(row['indices']))].append(row)
    rng=random.Random(seed);chosen=[]
    count=min(count,len(rows));n=count//len(groups)
    for _,values in sorted(groups.items()):chosen.extend(rng.sample(values,min(n,len(values))))
    rng.shuffle(chosen);return chosen


def check(bridge,state,rows,stage,details=False):
    standard=audit(bridge,state,rows,'answer')
    result={'answer':standard['answer_accuracy']}
    records={'standard':standard}
    if stage!='single':
        changed=audit(bridge,state,[changed_query(r) for r in rows],'answer')
        reversed_result=audit(bridge,state,[reversed_events(r) for r in rows],'answer')
        result['paired_query']=sum(a['answer_correct'] and b['answer_correct'] for a,b in zip(standard['details'],changed['details']))/len(rows)
        for n in (1,2):
            indices=[i for i,r in enumerate(rows) if len(r['indices'])==n]
            if indices:result[f'paired_reversal_{n}']=sum(standard['details'][i]['answer_correct'] and reversed_result['details'][i]['answer_correct'] for i in indices)/len(indices)
        records.update(changed_query=changed,reversed_events=reversed_result)
    if details:result['records']=records
    return result
