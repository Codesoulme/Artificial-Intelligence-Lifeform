"""Fixed-budget temporal meta-training from random weights, with no audit access."""
import argparse
import json
from pathlib import Path
import time
import torch
from .temporal import TemporalConfig,TemporalSPORA
from .temporal_tasks import events,queries
from .p4 import task_loss,evaluate
from .checkpoint import save_checkpoint


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--kind',choices=['split','shared'],default='split')
    p.add_argument('--warmup',type=int,default=200)
    p.add_argument('--steps-per-length',type=int,default=60)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    args=p.parse_args()
    directory=Path(args.run_dir)
    if Path(args.checkpoint).exists() or (directory/'config.json').exists():raise FileExistsError('Choose fresh run paths')
    if args.warmup<0 or args.steps_per_length<1:p.error('Invalid budget')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    generator=torch.Generator().manual_seed(910000+args.seed)
    config=TemporalConfig(temporal_kind=args.kind,memory_width=90 if args.kind=='split' else 127)
    model=TemporalSPORA(config).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    batch=4;segment=8
    directory.mkdir(parents=True,exist_ok=True)
    initial,_=model.initial_state(1)
    info={'arguments':vars(args),'config':model.config_dict(),'parameters':model.parameter_count(),
          'fast_parameters':sum(x.numel() for site in initial for x in site),
          'source':'random initialization','batch':batch,'segment':segment,
          'optimizer':{'name':'AdamW','lr':3e-4,'weight_decay':.01,'clip':1.},
          'loss':'equal group CE (current half known/half unknown, history, invalid, valid future) + 0.1 auxiliary'}
    (directory/'config.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();presentations=0;trusted_count=0;outer_steps=0
    with (directory/'metrics.jsonl').open('w',encoding='utf-8') as log:
        def record(row):
            row['seconds']=time.perf_counter()-started
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
        phases=[('warmup',8,args.warmup)]+[('curriculum',n,args.steps_per_length) for n in (8,16,32)]
        for phase,length,steps in phases:
            for step in range(steps):
                data=events(generator,batch,length,config,args.device)
                fast,velocity=model.initial_state(batch,True)
                optimizer.zero_grad(set_to_none=True)
                loss_value=0.
                for begin in range(0,length,segment):
                    end=min(begin+segment,length)
                    for i in range(begin,end):
                        fast,velocity=model.write_temporal(fast,velocity,data['keys'][:,i],data['values'][:,i],
                            data['times'][:,i],data['trusted'][:,i],differentiable=True)
                    loss=task_loss(model,fast,queries(data,end,config))/(length//segment)
                    if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite temporal objective')
                    loss.backward();loss_value+=float(loss.detach())
                    fast=[[x.detach().requires_grad_(True) for x in site] for site in fast]
                    velocity=[[x.detach() for x in site] for site in velocity]
                auxiliary=.1*model.auxiliary_loss();auxiliary.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();outer_steps+=1;presentations+=batch*length
                trusted_count+=int(data['trusted'].sum())
                if step==0 or (step+1)%20==0:
                    record({'phase':phase,'length':length,'step':step+1,'loss':loss_value,'gradient':float(gradient)})
            development=evaluate(model,32,length,1010000)
            record({'phase':'development','length':length,'scores':development})
            save_checkpoint(args.checkpoint,model,optimizer=optimizer,metadata={'stage':'P4 temporal foundation',
                'seed':args.seed,'completed_phase':phase,'completed_length':length,'origin':'random'})
        final={str(n):evaluate(model,32,n,1010000) for n in (8,16,32)}
        if args.device=='cuda':torch.cuda.synchronize()
        summary={'outer_steps':outer_steps,'event_presentations':presentations,
                 'bank_write_attempts':2*presentations,'nonzero_bank_writes':presentations+trusted_count,
                 'site_updates':len(model.memories)*(presentations+trusted_count),
                 'elapsed_seconds':time.perf_counter()-started,
                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
                 'development':final,'checkpoint':str(Path(args.checkpoint).resolve())}
        (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        record({'phase':'complete',**summary})


if __name__=='__main__':main()
