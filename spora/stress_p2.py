"""Locked-policy stress: repeated corrections without a helpful repeat."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .checkpoint import load_checkpoint
from .tasks import mappings
from .p2 import Event,write


def stress(model,policy,threshold,worlds=64,rounds=16):
    results=[]
    for mode in ('same_key','alternating_four'):
        generator=torch.Generator().manual_seed(620000)
        totals={'immediate':0,'after_gap':0,'unrelated':0,'final':0,'updates':0,
                'events':0,'untrusted_unchanged':0}
        for offset in range(0,worlds,8):
            batch=min(8,worlds-offset)
            keys,values=mappings(generator,batch,64,model.config,'cpu')
            fast,velocity=model.initial_state(batch)
            def accept(event):
                nonlocal fast,velocity
                fast,velocity,steps=write(model,fast,velocity,event,policy,threshold)
                totals['updates']+=float(steps.sum())
                totals['events']+=batch*int(bool(event.trusted.any()))
            one=torch.ones(batch);zero=torch.zeros(batch)
            for i in range(64):accept(Event(keys[:,i],values[:,i],one,'new'))
            for revision in range(rounds):
                pos=0 if mode=='same_key' else revision%4
                shift=torch.randint(1,model.config.value_count,(batch,),generator=generator)
                values[:,pos]=(values[:,pos]+shift)%model.config.value_count
                accept(Event(keys[:,pos],values[:,pos],one,'correction'))
                with torch.no_grad():
                    totals['immediate']+=int((model(keys[:,pos],fast).argmax(-1)==values[:,pos]).sum())
                snapshot=[[p.detach().clone() for p in site] for site in fast+velocity]
                accept(Event(keys[:,pos],(values[:,pos]+1)%model.config.value_count,zero,'untrusted'))
                same=all(torch.equal(a,b) for sa,sb in zip(snapshot,fast+velocity) for a,b in zip(sa,sb))
                totals['untrusted_unchanged']+=batch*int(same)
                # Eight intervening, newly revised unrelated facts; never repeat the target.
                for gap in range(8):
                    j=4+(revision*8+gap)%60
                    shift=torch.randint(1,model.config.value_count,(batch,),generator=generator)
                    values[:,j]=(values[:,j]+shift)%model.config.value_count
                    accept(Event(keys[:,j],values[:,j],one,'interference'))
                with torch.no_grad():
                    pred=model.predict_many(keys,fast).argmax(-1)
                totals['after_gap']+=int((pred[:,pos]==values[:,pos]).sum())
                totals['unrelated']+=int((pred[:,4:]==values[:,4:]).sum())
            totals['final']+=int((pred==values).sum())
        results.append({'mode':mode,'worlds':worlds,'rounds':rounds,
                        'immediate':totals['immediate']/(worlds*rounds),
                        'after_gap':totals['after_gap']/(worlds*rounds),
                        'unrelated':totals['unrelated']/(worlds*rounds*60),
                        'final':totals['final']/(worlds*64),
                        'untrusted_unchanged':totals['untrusted_unchanged']/(worlds*rounds),
                        'inner_updates_per_trusted_event':totals['updates']/totals['events']})
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--selection',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    target=Path(args.output)
    if target.exists():raise FileExistsError('Preserve previous stress result')
    development=json.loads(Path(args.selection).read_text(encoding='utf-8'))
    digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if development['checkpoint_sha256']!=digest:raise ValueError('Checkpoint changed')
    torch.set_num_threads(4)
    model,_,_,_=load_checkpoint(args.checkpoint)
    result={'checkpoint':args.checkpoint,'sha256':digest,'selected':development['selected'],
            'results':stress(model,**development['selected']),
            'limits':'Current-value queries only; finite vocabulary; no hyperparameter selection on this stream'}
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
