"""Measure dispatcher-covered arithmetic; this is not a hardware cost model."""
import json
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode
from spora.model import SPORA,ModelConfig
from spora.tasks import mappings,absorb
from spora.p2 import stream,world_loss,geometry_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='runs/p2-compute-counter.json')
    parser.add_argument('--method',choices=['dispatch','profiler'],default='dispatch')
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():raise FileExistsError('Preserve profile')
    torch.set_num_threads(2);rows={}
    for kind in ('mlp','delta'):
        torch.manual_seed(7)
        model=SPORA(ModelConfig(memory_kind=kind,memory_width=90 if kind=='delta' else 64))
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        generator=torch.Generator().manual_seed(520007)
        costs={}
        for length in ('warmup',8,16,32,64):
            if length=='warmup':keys,values=mappings(generator,4,8,model.config,'cpu')
            else:events,keys,_,_=stream(generator,4,length,model.config,'cpu')
            counter=FlopCounterMode(display=False) if args.method=='dispatch' else torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True)
            with counter as profile:
                optimizer.zero_grad(set_to_none=True)
                if length=='warmup':
                    fast,velocity=absorb(model,keys,values,differentiable=True)
                    loss=F.cross_entropy(model.predict_many(keys,fast).flatten(0,1),values.flatten())+.1*model.auxiliary_loss()
                    loss.backward()
                else:
                    fast,velocity=model.initial_state(4,True)
                    world_loss(model,events,keys,fast,velocity,'one',.8)
                    (.1*model.auxiliary_loss()+0.*geometry_loss(model)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            if args.method=='dispatch':
                operators={str(op):int(value) for op,value in profile.get_flop_counts()['Global'].items()}
            else:
                operators={event.key:int(event.flops) for event in profile.key_averages() if event.flops}
            costs[str(length)]={'covered_flops':sum(operators.values()),'operators':operators}
            print(json.dumps({'kind':kind,'length':length,'covered_flops':sum(operators.values())}),flush=True)
        total=300*costs['warmup']['covered_flops']+40*sum(costs[str(k)]['covered_flops'] for k in (8,16,32,64))
        rows[kind]={'per_step':costs,'matched_event_run_covered_flops':total}
    result={'batch_size':4,'method':args.method,'rows':rows,'limitations':[
        'PyTorch counts supported arithmetic only; nonlinear ops, memory traffic and kernel overhead are incomplete',
        'CPU trace used for arithmetic shapes; concurrent wall times are not equal-compute evidence',
        'Same counts assumed per shape; clipped-gradient numeric values do not change tensor shapes']}
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
