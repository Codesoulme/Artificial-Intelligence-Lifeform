"""No-consolidation and exact-rebase controls over the same fixed audit stream."""
import json
from pathlib import Path
import torch
from spora.consolidate_p3 import chain
from spora.checkpoint import load_checkpoint
from spora.tasks import mappings
from spora.p2 import Event,write


def main():
    torch.set_num_threads(2)
    output=Path('runs/p3-controls')
    if output.exists():raise FileExistsError('Preserve controls')
    output.mkdir()
    results=[]
    for seed in (7,11,23):
        checkpoint=f'checkpoints/p2-v1-delta-seed{seed}.pt'
        dest=output/f'seed{seed}';dest.mkdir()
        exact=chain(checkpoint,910000,4,3,0,{'name':'exact_rebase'},'direct',dest,'cpu',True,True)
        model,_,_,_=load_checkpoint(checkpoint)
        generator=torch.Generator().manual_seed(910000)
        no_transfer=[]
        for world in range(4):
            keys,values=mappings(generator,1,24,model.config,'cpu')
            fast,velocity=model.initial_state(1)
            for stage in range(1,4):
                for i in range((stage-1)*8,stage*8):
                    event=Event(keys[:,i],values[:,i],torch.ones(1),'new')
                    fast,velocity,_=write(model,fast,velocity,event,'one')
                reset,_=model.initial_state(1)
                with torch.no_grad():
                    kept=model.predict_many(keys[:,:stage*8],fast).argmax(-1)
                    erased=model.predict_many(keys[:,:stage*8],reset).argmax(-1)
                no_transfer.append({'world':world,'stage':stage,
                    'retained_F':float((kept==values[:,:stage*8]).float().mean()),
                    'reset_F':float((erased==values[:,:stage*8]).float().mean())})
        results.append({'foundation_seed':seed,'exact':exact,'no_consolidation':no_transfer})
    (output/'controls.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'saved':str(output/'controls.json')}),flush=True)


if __name__=='__main__':main()
