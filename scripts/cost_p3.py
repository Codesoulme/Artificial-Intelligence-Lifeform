"""Count incremental arithmetic for a direct or residual single-event write."""
import copy
import json
from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
from spora.checkpoint import load_checkpoint
from spora.p2 import Event,write


def main():
    path=Path('runs/p3-write-cost.json')
    if path.exists():raise FileExistsError('Preserve cost result')
    torch.set_num_threads(2)
    foundation,fast,velocity,_=load_checkpoint('checkpoints/individual-p2-candidate.pt')
    event=Event(torch.tensor([12]),torch.tensor([3]),torch.ones(1),'correction')
    rows=[]
    for residual in (False,True):
        model=copy.deepcopy(foundation);model.config.residual_write=residual
        with FlopCounterMode(display=False) as count:
            ff,vv,steps=write(model,fast,velocity,event,'one')
        rows.append({'writer':'residual' if residual else 'direct',
                     'covered_flops':count.get_total_flops(),'internal_updates':int(steps.item()),
                     'operators':{str(k):v for k,v in count.get_flop_counts()['Global'].items()}})
    with FlopCounterMode(display=False) as count:
        with torch.no_grad():foundation(event.key,fast)
    result={'single_individual':True,'rows':rows,'read_covered_flops':count.get_total_flops(),
            'limits':'Matrix arithmetic only; extra nonlinear operations and memory traffic are not fully counted; no wall-clock claim'}
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
