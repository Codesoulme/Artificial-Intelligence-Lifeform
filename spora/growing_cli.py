"""Explicit growing-individual lifecycle, serialized writes, no silent growth."""
import argparse,json
from pathlib import Path
import torch
from .growing import load,save,grow,observe,query,scalars,migrate,capacity

def mutate(path,operation,**kwargs):
    path=Path(path);lock=path.with_suffix('.lock');handle=lock.open('x')
    try:
        with handle:
            model,state,meta=load(path)
            if operation=='learn':state=observe(model,state,**kwargs)
            elif operation=='grow':state=grow(model,state,**kwargs)
            else:raise ValueError('Unknown mutation')
            save(path,model,state,meta)
    finally:lock.unlink()
    return {'observations':state.cursor,'capacity':capacity(model,state),'scalars':scalars(state)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--individual',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    adopt=sub.add_parser('adopt');adopt.add_argument('--source',required=True)
    learn=sub.add_parser('learn');learn.add_argument('key',type=int);learn.add_argument('value',type=int)
    learn.add_argument('--trusted',type=int,choices=[0,1],default=1)
    growth=sub.add_parser('grow');growth.add_argument('--pages',type=int,default=1)
    ask=sub.add_parser('ask');ask.add_argument('key',type=int);ask.add_argument('--time',type=int)
    sub.add_parser('info');a=p.parse_args();torch.set_num_threads(2)
    if a.command=='adopt':
        path=Path(a.individual);path.parent.mkdir(parents=True,exist_ok=True);lock=path.with_suffix('.lock');handle=lock.open('x')
        try:
            with handle:model,state=migrate(a.source,path)
        finally:lock.unlink()
        result={'adopted':str(path.resolve()),'observations':state.cursor}
    elif a.command=='learn':result=mutate(a.individual,'learn',key=a.key,value=a.value,trusted=a.trusted)
    elif a.command=='grow':result=mutate(a.individual,'grow',pages=a.pages)
    else:
        model,state,meta=load(a.individual)
        if a.command=='info':result={'observations':state.cursor,'capacity':capacity(model,state),'scalars':scalars(state),'metadata':meta}
        else:
            with torch.no_grad():prob=query(model,state,torch.tensor([a.key]),torch.tensor([int(a.time is not None)]),torch.tensor([a.time or 0])).softmax(-1)[0]
            answer=int(prob.argmax());result={'value':None if answer==16 else answer,'unknown':answer==16,'probability':float(prob[answer])}
    print(json.dumps(result))

if __name__=='__main__':main()
