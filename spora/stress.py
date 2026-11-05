"""Exploratory capacity and correction probes, separate from primary audit."""
import argparse
import json
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .tasks import mappings, absorb


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():
        raise FileExistsError('Stress output already exists')
    torch.set_num_threads(4)
    model,_,_,_=load_checkpoint(args.checkpoint)
    generator=torch.Generator().manual_seed(300000)
    totals={size:{'all':0.,'oldest8':0.,'latest8':0.} for size in (8,16,32,64)}
    corrections={'updated':0.,'unaffected':0.}
    worlds=32
    for offset in range(0,worlds,8):
        keys,values=mappings(generator,8,64,model.config,'cpu')
        fast,velocity=model.initial_state(8)
        for index in range(64):
            fast,velocity=model.write_event(fast,velocity,keys[:,index],values[:,index])
            size=index+1
            if size in totals:
                with torch.no_grad():
                    matches=(model.predict_many(keys[:,:size],fast).argmax(-1)==values[:,:size]).float()
                totals[size]['all']+=matches.mean(1).sum().item()/worlds
                totals[size]['oldest8']+=matches[:,:8].mean(1).sum().item()/worlds
                totals[size]['latest8']+=matches[:,-8:].mean(1).sum().item()/worlds
        fast,velocity=absorb(model,keys[:,:8],values[:,:8])
        revised=(values[:,:1]+1)%model.config.value_count
        fast,velocity=absorb(model,keys[:,:1],revised,fast,velocity)
        with torch.no_grad():
            predicted=model.predict_many(keys[:,:8],fast).argmax(-1)
        corrections['updated']+=(predicted[:,:1]==revised).float().mean(1).sum().item()/worlds
        corrections['unaffected']+=(predicted[:,1:]==values[:,1:8]).float().mean(1).sum().item()/worlds
    report={'worlds':worlds,'seed':300000,'capacity':totals,'single_write_correction':corrections,
            'interpretation':'Exploratory stress probe, not used for selection or retraining.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
