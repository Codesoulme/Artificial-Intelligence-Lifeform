"""Finite temporal training, with development separated from later audit."""
import argparse,json,sys,time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.temporal_tasks import events,queries
from spora.p4 import task_loss,evaluate
from spora.checkpoint import save_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['rls90','delta110','orthogonal90'],required=True)
    p.add_argument('--seed',type=int,default=7);args=p.parse_args()
    prefix='p5b' if args.variant=='orthogonal90' else 'p5'
    name=f'{prefix}-{args.variant}-seed{args.seed}';directory=ROOT/'runs'/name;checkpoint=ROOT/'checkpoints'/f'{name}.pt'
    if directory.exists() or checkpoint.exists():raise FileExistsError(name)
    directory.mkdir();torch.set_num_threads(4);torch.manual_seed(args.seed)
    plan=json.loads((ROOT/'configs'/('p5b-plan.json' if prefix=='p5b' else 'p5-plan.json')).read_text())
    variant={'width':90,'writer':'rls'} if prefix=='p5b' else next(v for v in plan['candidates'] if v['name']==args.variant)
    config=TemporalConfig(memory_width=variant['width'],history_writer=variant['writer'],current_address='orthogonal' if prefix=='p5b' else 'learned')
    model=TemporalSPORA(config).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    development=1510000 if prefix=='p5b' else 1210000
    generator=torch.Generator().manual_seed(development-10000+args.seed);torch.cuda.reset_peak_memory_stats();start=time.perf_counter();seen=0
    (directory/'config.json').write_text(json.dumps({'plan':plan,'config':model.config_dict(),'parameters':model.parameter_count()},indent=2))
    log=(directory/'metrics.jsonl').open('w')
    def record(row):
        row['seconds']=time.perf_counter()-start;log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    phases=[(8,200)]+[(n,100) for n in (8,16,32,64)]
    for length,steps in phases:
        for step in range(steps):
            data=events(generator,4,length,config,'cuda')
            if prefix=='p5b' and step%2:
                data['keys']=torch.stack([torch.randperm(64,generator=generator)[:length] for _ in range(4)]).cuda()
                data['trusted']=(torch.rand(4,length,generator=generator)>.25).float().cuda()
            fast,velocity=model.initial_state(4,True)
            optimizer.zero_grad(set_to_none=True);loss_value=0
            for begin in range(0,length,8):
                end=begin+8
                for i in range(begin,end):
                    fast,velocity=model.write_temporal(fast,velocity,data['keys'][:,i],data['values'][:,i],data['times'][:,i],data['trusted'][:,i],True)
                group=queries(data,end,config)
                if prefix=='p5b' and step%2:
                    # All observed symbols, including trust=0, plus truly unseen symbols.
                    ks=data['keys'][:,:end];targets=torch.where(data['trusted'][:,:end].bool(),data['values'][:,:end],torch.full_like(ks,16))
                    group['current']=(ks,torch.zeros_like(ks),torch.zeros_like(ks),targets)
                # Vary negative key offsets instead of learning only a +1 negative form.
                k,m,t,y=group['invalid'];shift=torch.randint(1,64,k.shape,generator=generator).to(k.device)
                group['invalid']=((data['keys'][:,:end]+shift)%64,m,t,y)
                loss=task_loss(model,fast,group)/(length//8);loss.backward();loss_value+=float(loss.detach())
                fast=[[x.detach().requires_grad_(True) for x in s] for s in fast]
                velocity=[[x.detach() for x in s] for s in velocity]
            (.1*model.auxiliary_loss()).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();seen+=length*4
            if (step+1)%25==0:record({'length':length,'step':step+1,'loss':loss_value})
        record({'development':length,'metrics':evaluate(model,32,length,development)})
    final={str(n):evaluate(model,32,n,development) for n in (8,16,32,64)}
    save_checkpoint(checkpoint,model,metadata={'stage':'P5 temporal foundation','seed':args.seed,'variant':args.variant,'origin':'random'})
    summary={'development':final,'seconds':time.perf_counter()-start,'event_presentations':seen,'steps':600,
             'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'checkpoint':str(checkpoint)}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2));record(summary);log.close()

if __name__=='__main__':main()
