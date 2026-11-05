"""Explicit same-key absent-time checks plus empty allocated pages."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.checkpoint import load_checkpoint
from spora.temporal_tasks import events
from spora.p4 import consume
from spora.growing import new_state,grow,observe,query

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    output=ROOT/'runs'/f'p6-negatives-seed{a.seed}.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2);model,_,_,_=load_checkpoint(ROOT/'checkpoints'/f'p5b-orthogonal90-seed{a.seed}.pt')
    gen=torch.Generator().manual_seed(1820000)
    correct=total=0
    for start in range(0,64,4):
        r=events(gen,4,64,model.config,'cpu');fast,_=consume(model,r)
        keys=[];times=[]
        for row in range(4):
            kk=[];tt=[]
            for i in range(64):
                # Keep the actual entity, ask at a time belonging to another entity.
                key=r['keys'][row,i];possible=torch.nonzero(r['keys'][row]!=key).flatten()
                j=int(possible[i%len(possible)])
                kk.append(int(key));tt.append(j)
            keys.append(kk);times.append(tt)
        k=torch.tensor(keys);t=torch.tensor(times)
        predictions=model.predict_many(k,fast,torch.ones_like(k),t).argmax(-1)
        correct+=int((predictions==16).sum());total+=predictions.numel()
    state=new_state(model)
    for time in range(64):state=observe(model,state,time%4,time%16,1)
    state=grow(model,state,pages=3)
    k=torch.arange(64).repeat(3);t=torch.arange(64,256)
    unknown=query(model,state,k,torch.ones_like(k),t).argmax(-1)
    result={'seed':a.seed,'same_key_wrong_time':{'correct':correct,'total':total,'accuracy':correct/total},
            'empty_allocated_pages':{'correct':int((unknown==16).sum()),'total':unknown.numel(),'accuracy':float((unknown==16).float().mean())}}
    result['passed']=correct/total>=.95 and result['empty_allocated_pages']['accuracy']>=.95
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
