"""Explicit temporal individual lifecycle; every learn call is one observation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .checkpoint import load_checkpoint,save_checkpoint
from .temporal import TemporalSPORA


def require_temporal(model,fast):
    if not isinstance(model,TemporalSPORA):raise ValueError('A temporal foundation is required')
    if fast[0][0].shape[0]!=1:raise ValueError('Expected one individual')


def create(path,foundation):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock')
    handle=lock.open('x')
    try:
        with handle:
            if path.exists():raise FileExistsError('Individual exists; choose another path')
            model,fast,_,_=load_checkpoint(foundation)
            require_temporal(model,fast)
            fast,velocity=model.initial_state(1)
            digest=hashlib.sha256(Path(foundation).read_bytes()).hexdigest()
            save_checkpoint(path,model,fast,velocity,cursor=0,
                metadata={'stage':'P4 temporal research individual','parent_sha256':digest})
    finally:lock.unlink()
    return {'individual':str(path.resolve()),'observations':0}


def learn(path,key,value,trusted):
    path=Path(path);lock=path.with_suffix('.lock')
    handle=lock.open('x')
    try:
        with handle:
            model,fast,velocity,data=load_checkpoint(path)
            require_temporal(model,fast)
            cursor=data['cursor']
            if cursor>=model.config.time_count:
                raise ValueError('Time capacity exhausted; no wraparound or overwrite is allowed')
            fast,velocity=model.write_temporal(fast,velocity,torch.tensor([key]),torch.tensor([value]),
                torch.tensor([cursor]),torch.tensor([trusted],dtype=torch.float32))
            save_checkpoint(path,model,fast,velocity,cursor=cursor+1,metadata=data['metadata'])
    finally:lock.unlink()
    return {'event_time':cursor,'observations':cursor+1,'trusted':trusted,
            'nonzero_bank_writes':1+int(trusted),'saved':str(path.resolve())}


def main():
    p=argparse.ArgumentParser(description='Finite temporal symbolic learner; timestamps 0 through 63')
    p.add_argument('--individual',required=True)
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('create');c.add_argument('--foundation',required=True)
    l=sub.add_parser('learn');l.add_argument('key',type=int);l.add_argument('value',type=int)
    l.add_argument('--trusted',type=int,choices=[0,1],default=1)
    a=sub.add_parser('ask');a.add_argument('key',type=int);a.add_argument('--time',type=int)
    sub.add_parser('info')
    args=p.parse_args();torch.set_num_threads(2)
    if args.command=='create':result=create(args.individual,args.foundation)
    elif args.command=='learn':result=learn(args.individual,args.key,args.value,args.trusted)
    else:
        model,fast,_,data=load_checkpoint(args.individual);require_temporal(model,fast)
        if args.command=='info':
            result={'parameters':model.parameter_count(),'observations':data['cursor'],
                    'temporal_kind':model.config.temporal_kind,'time_capacity':model.config.time_count,
                    'stage':data['metadata'].get('stage')}
        else:
            mode=int(args.time is not None)
            with torch.no_grad():
                prob=model(torch.tensor([args.key]),fast,torch.tensor([mode]),torch.tensor([args.time or 0])).softmax(-1)[0]
            answer=int(prob.argmax())
            result={'mode':'event' if mode else 'current','key':args.key,'time':args.time,
                    'value':None if answer==model.config.unknown else answer,
                    'unknown':answer==model.config.unknown,'probability':float(prob[answer])}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
