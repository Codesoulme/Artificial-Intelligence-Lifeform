"""Finite P2 curriculum; outer updates never occur inside a persistent world."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .model import SPORA,ModelConfig
from .checkpoint import load_checkpoint,save_checkpoint
from .tasks import mappings,absorb
from .p2 import stream,world_loss,geometry_loss,evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--parent')
    parser.add_argument('--kind',choices=['mlp','delta'],default='mlp')
    parser.add_argument('--policy',choices=['one','two','adaptive-max4'],default='one')
    parser.add_argument('--threshold',type=float,default=.8)
    parser.add_argument('--steps-per-length',type=int,default=40)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--geometry',type=float,default=0.)
    parser.add_argument('--short-control',action='store_true',
                        help='Same parent and number of worlds, retain old 8-event task')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    directory=Path(args.run_dir)
    if Path(args.checkpoint).exists() or (directory/'config.json').exists():
        raise FileExistsError('Preserve previous run; choose new paths')
    if args.steps_per_length<1 or args.batch_size<1 or args.warmup<0:
        parser.error('Invalid training budget')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    generator=torch.Generator().manual_seed(args.seed+520000)
    if args.parent:
        model,_,_,_=load_checkpoint(args.parent,args.device)
        if model.config.memory_kind!=args.kind:raise ValueError('Parent architecture mismatch')
    else:
        # 2*90*90=16,200 fast parameters vs MLP 16,640 (2.64% less).
        config=ModelConfig(memory_kind=args.kind,memory_width=90 if args.kind=='delta' else 64)
        model=SPORA(config).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4 if args.parent else 3e-4,weight_decay=.01)
    directory.mkdir(parents=True,exist_ok=True)
    fast,_=model.initial_state(1)
    info={'arguments':vars(args),'config':model.config_dict(),'parameters':model.parameter_count(),
          'fast_parameters':sum(p.numel() for site in fast for p in site),
          'source':'P1 lineage continuation' if args.parent else 'random initialization'}
    (directory/'config.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();event_count=0;inner_updates=0;outer_steps=0
    log=(directory/'metrics.jsonl').open('a',encoding='utf-8')
    def record(row):
        row['seconds']=time.perf_counter()-started
        log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    for step in range(args.warmup):
        keys,values=mappings(generator,args.batch_size,8,model.config,args.device)
        optimizer.zero_grad(set_to_none=True)
        fast,velocity=absorb(model,keys,values,differentiable=True)
        loss=F.cross_entropy(model.predict_many(keys,fast).flatten(0,1),values.flatten())+.1*model.auxiliary_loss()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();outer_steps+=1;event_count+=8*args.batch_size;inner_updates+=8*args.batch_size
        if (step+1)%50==0:record({'phase':'warmup','step':step+1,'loss':float(loss.detach())})
    model.train()
    for pairs in (8,16,32,64):
        for step in range(args.steps_per_length):
            events,keys,original,current=stream(generator,args.batch_size,8 if args.short_control else pairs,
                                              model.config,args.device,corrections=0 if args.short_control else 4)
            fast,velocity=model.initial_state(args.batch_size,True)
            optimizer.zero_grad(set_to_none=True)
            loss,updates=world_loss(model,events,keys,fast,velocity,args.policy,args.threshold)
            auxiliary=.1*model.auxiliary_loss()+args.geometry*geometry_loss(model)
            auxiliary.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();outer_steps+=1;event_count+=len(events)*args.batch_size;inner_updates+=updates
            if step==0 or (step+1)%10==0:
                record({'phase':'curriculum','pairs':pairs,'step':step+1,'loss':loss,'gradient_norm':float(grad)})
        development=evaluate(model,worlds=16,pairs=pairs,seed=510000+pairs,policy=args.policy,threshold=args.threshold)
        record({'phase':'development','pairs':pairs,'scores':development})
        save_checkpoint(args.checkpoint,model,optimizer=optimizer,metadata={
            'stage':'P2 curriculum','seed':args.seed,'completed_pairs':pairs,
            'policy':args.policy,'parent':args.parent or 'random','geometry':args.geometry})
    if args.device=='cuda':torch.cuda.synchronize()
    summary={'outer_steps':outer_steps,'event_presentations':event_count,'inner_updates':inner_updates,
             'elapsed_seconds':time.perf_counter()-started,
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
             'checkpoint':str(Path(args.checkpoint).resolve()),'development_final':development}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    log.close();print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
