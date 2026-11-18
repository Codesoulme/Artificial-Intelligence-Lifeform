"""Audit trained neural history plus focus; no base-language inference or raw answers in state."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from spora.event_focus import (enable_event_focus,observe_focused,bind_event_focus,
                               clear_event_focus,read_focused_event,focused_memory_args)
from spora.growing import new_state,query,save,load,grow
from spora.internal_memory import (WriteControl,NoveltyWriteControl,attach,enable_event_key_index,
                                    enable_independent_provenance,observe_sourced)
from spora.integrated_checkpoint import _digest
from spora.state_codec import flatten
from spora.temporal import TemporalSPORA,TemporalConfig

SOURCE=ROOT/'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
OUT=ROOT/'runs/spora-event-focus-audit-20261118'


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with path.open('x',encoding='utf8') as f: json.dump(value,f,ensure_ascii=False,indent=2)


def fingerprint(model,state):
    schema,tensors=flatten(model,state)
    return _digest(schema,tensors)


@torch.no_grad()
def measure(model,state):
    selection=read_focused_event(model,state)
    result={'selection':selection,'focus':state.internal.event_focus.tolist(),'cursor':state.cursor}
    if selection['status']=='experimental_candidate':
        keys=torch.tensor([selection['key']]);mode=torch.ones_like(keys)
        logits=query(model,state,keys,mode,torch.full_like(keys,selection['event_time']))
        result.update(value=int(logits.argmax(-1)[0]),logits=logits[0].tolist())
        controls,_=focused_memory_args(model,state)
        result['slot']=controls['memory_slots'].tolist()
        result['address_shapes']=[list(a.shape) for a in controls['memory_addresses']]
    return result


def fresh():
    payload=torch.load(SOURCE,map_location='cpu',weights_only=True)
    if _digest(payload['metadata'],payload['tensors'])!=payload['integrity_sha256']:
        raise ValueError('Parent integrity mismatch')
    schema=payload['metadata']['individual_schema']
    model=TemporalSPORA(TemporalConfig(**schema['config']))
    prefix='individual/model/'
    core={name:payload['tensors'][prefix+name] for name in model.state_dict()}
    model.load_state_dict(core,strict=True)
    config=dict(schema['write_control'])
    controller=(NoveltyWriteControl if config.pop('kind',None)=='novelty' else WriteControl)(**config)
    controller.load_state_dict({name:payload['tensors'][prefix+'write_control.'+name] for name in controller.state_dict()},strict=True)
    state=attach(model,new_state(model),controller)
    state=enable_independent_provenance(model,enable_event_key_index(model,state))
    state=enable_event_focus(model,state)
    model.eval();model.requires_grad_(False)
    return model,state


@torch.no_grad()
def create():
    if OUT.exists(): raise FileExistsError(OUT)
    torch.set_num_threads(2);torch.manual_seed(55781)
    started=time.perf_counter()
    paths=[SOURCE,Path(__file__)]+[ROOT/'spora'/n for n in
        ('event_focus.py','internal_memory.py','state_codec.py','growing.py','integrated_checkpoint.py')]
    hashes={str(p):sha(p) for p in paths}
    model,state=fresh()
    OUT.mkdir()
    original_weights={n:p.detach().clone() for n,p in model.state_dict().items()}
    results=[];artifacts={}
    def capture(label,current,expected_key,expected_event,expected_value):
        before=fingerprint(model,current)
        record=measure(model,current)
        record.update(label=label,expected={'key':expected_key,'event':expected_event,'value':expected_value})
        record['correct']=(record['selection'].get('key')==expected_key and
                           record['selection'].get('event_time')==expected_event and record.get('value')==expected_value)
        if fingerprint(model,current)!=before: raise ValueError('Read mutated state')
        path=OUT/(label+'.pt');save(path,model,current)
        artifacts[path.name]={'sha256':sha(path),'state_digest':before}
        results.append(record)
    empty=read_focused_event(model,state)
    state,a=observe_focused(model,state,2,3,'direct',1.)
    capture('object_a',state,2,0,3)
    state,b=observe_focused(model,state,5,7,'direct',1.)
    capture('object_b',state,5,1,7)
    # Binding an older observation must still select its stored historical value.
    state=bind_event_focus(model,state,2,0)
    capture('return_a',state,2,0,3)
    state,c=observe_sourced(model,state,2,9,'direct',1.)
    capture('background_correction_pins_old',state,2,0,3)
    state=bind_event_focus(model,state,2,c['history_time'])
    capture('select_corrected_a',state,2,2,9)
    state,d=observe_focused(model,state,5,4,'direct',1.)
    capture('foreground_correction_b',state,5,3,4)
    state,e=observe_focused(model,state,2,12,'simulation',1.)
    capture('simulation_keeps_b',state,5,3,4)
    negatives={}
    for label,key,event in (('wrong_object',5,0),('future',2,state.cursor),('simulation',2,4)):
        try: bind_event_focus(model,state,key,event)
        except ValueError as exc: negatives[label]={'rejected':True,'reason':str(exc)}
        else: negatives[label]={'rejected':False}
    cleared=clear_event_focus(model,state)
    negatives['clear']={'rejected':read_focused_event(model,cleared)['status']=='abstain'}
    damaged=copy.deepcopy(state)
    for site in damaged.pages[0]: site[0].zero_()
    negatives['unreadable_history']={'rejected':read_focused_event(model,damaged)['status']=='abstain',
                                   'result':read_focused_event(model,damaged)}
    grown=grow(model,state)
    capture('growth_keeps_b',grown,5,3,4)
    weights_unchanged=all(torch.equal(p,original_weights[n]) for n,p in model.state_dict().items())
    if not weights_unchanged or any(sha(p)!=hashes[str(p)] for p in paths):
        raise ValueError('Protected weights or sources changed')
    write(OUT/'result.json',{'source_sha256':hashes,'records':results,'artifacts':artifacts,
          'decisions':[a,b,c,d,e],'negative_checks':negatives,'initial':empty,'seconds':time.perf_counter()-started,
          'correct':sum(r['correct'] for r in results),'total':len(results),'weights_unchanged':weights_unchanged,
          'scope':'Frozen trained memory core and write controller from integrated DeepSeek candidate, fresh isolated history.',
          'limits':['Base language model, reader and output adapter are not exercised.',
                    'Auxiliary skill/time/semantic modules and old individual history are not transplanted into this fixture.',
                    'Finite keys and values explicitly provided; no learned reference resolution claim.',
                    'No formal individual checkpoint is modified.']})
    print(json.dumps({'correct':sum(r['correct'] for r in results),'total':len(results),'negative_checks':negatives}),flush=True)


def verify():
    target=OUT/'reload.json'
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    result=json.loads((OUT/'result.json').read_text(encoding='utf8'))
    for p,digest in result['source_sha256'].items():
        if sha(Path(p))!=digest: raise ValueError('Source changed')
    records=[]
    for reference in result['records']:
        name=reference['label']+'.pt';artifact=result['artifacts'][name]
        if sha(OUT/name)!=artifact['sha256']: raise ValueError('Saved state changed')
        model,state,_=load(OUT/name)
        if fingerprint(model,state)!=artifact['state_digest']: raise ValueError('Restored state changed')
        actual=measure(model,state)
        if any(actual[k]!=reference[k] for k in actual): raise ValueError('Readback mismatch: '+name)
        records.append({'label':reference['label'],'exact_readback':True,'value':actual.get('value')})
    write(target,{'verified':records,'scope':'Separate process reload of actual neural core and focus; no language inference'})
    print(json.dumps(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('create','verify'))
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.out
    {'create':create,'verify':verify}[args.stage]()
