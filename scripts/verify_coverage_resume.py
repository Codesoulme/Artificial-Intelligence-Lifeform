"""Two independent restores replay the P27 sampler and optimizer, read-only."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tokenizers import Tokenizer
from spora.conversation import restore_conversation
from spora.conversation_optimization import backward_supervised
from spora.growing import load
from spora.language_training import frozen_digest, interface_state
from spora.training_mix import choose_source
from scripts.train_conversation_v7 import ROOT, sha, load_rows, padded, text_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=20)
    args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    if not 1<=args.steps<=20:raise ValueError('Bounded replay required')
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));data=ROOT/plan['data'];curriculum=ROOT/plan['curriculum']
    path=ROOT/args.checkpoint;before_file=sha(path);payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['plan_sha256']!=sha(ROOT/args.plan):raise ValueError('Wrong plan')
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Parent changed')
    torch.set_num_threads(2)
    tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'));pad=tokenizer.token_to_id('<|pad|>')
    stream=np.memmap(data/'pretrain-train.bin',dtype=np.uint16,mode='r')
    sources={'teacher':load_rows(curriculum/'teacher-tokens.jsonl',plan['length']),
        'foundation':load_rows(curriculum/'foundations-tokens.jsonl',plan['length']),
        'dialogue':load_rows(data/'dialogue-train-tokens.jsonl',plan['length'])}
    focus=json.loads((data/'focus-tokens.json').read_text('utf-8'))
    outputs=[];traces=[];checks=[]
    for attempt in range(2):
        organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
        model=restore_conversation(organism,payload,plan['parent_sha256'],sha(data/'tokenizer.json'),'cuda')
        optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
        optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
        rng=random.Random();rng.setstate(payload['python_rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        counts=copy.deepcopy(payload['counts']);focus_cursor=payload['focus_cursor'];trace=[]
        for step in range(payload['steps']+1,payload['steps']+args.steps+1):
            fraction=min(step/plan['warmup'],1.) if step<=plan['warmup'] else .1+.9*.5*(1+math.cos(math.pi*(step-plan['warmup'])/(plan['schedule_steps']-plan['warmup'])))
            for group in optimizer.param_groups:group['lr']=plan['learning_rate']*fraction
            kind=choose_source(plan['target_shares'],counts)
            if kind=='pretrain':x,y=text_batch(stream,rng,plan['batch'],plan['length'],'cuda')
            else:
                rows=[rng.choice(sources[kind]) for _ in range(plan['batch'])]
                length=min(plan['length'],32*math.ceil((max(len(r['tokens']) for r in rows)-1)/32))
                x,y=padded(rows,length,pad,'cuda')
            focus_n=0
            if step%plan['focus_every']==0:
                seq=focus[focus_cursor%len(focus)][:x.shape[1]+1];focus_cursor+=1
                fx,fy=padded([dict(tokens=seq,mask=[True]*len(seq))],x.shape[1],pad,'cuda')
                x[0]=fx[0];y[0]=fy[0];focus_n=int((fy!=-100).sum())
            digest=hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
            optimizer.zero_grad(set_to_none=True);loss=backward_supervised(model,state,x,y,plan['microbatch'])
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
            n=int((y!=-100).sum());key=kind+'_source_targets';counts[key]=counts.get(key,0)+n-focus_n
            trace.append(dict(step=step,kind=kind,batch_sha256=digest,loss=float(loss.detach()),source_targets=n-focus_n))
        outputs.append(interface_state(model));traces.append(trace);checks.append(before==frozen_digest(organism,state))
        del optimizer,model,organism,state
    exact=all(torch.equal(v,outputs[1][k]) for k,v in outputs[0].items())
    report=dict(checkpoint_sha256=before_file,source_step=payload['steps'],replayed_steps=args.steps,
        traces_equal=traces[0]==traces[1],replayed_weights_exact=exact,
        maximum_tensor_difference=max(float((v-outputs[1][k]).abs().max()) for k,v in outputs[0].items()),
        old_individual_exact=all(checks),checkpoint_unchanged=sha(path)==before_file,trace=traces[0],
        scope='Two independent checkpoint restores replay identical token-mixture/optimizer steps; not comparison with a separately uninterrupted long run.')
    report['passed']=all(report[k] for k in ('traces_equal','replayed_weights_exact','old_individual_exact','checkpoint_unchanged'))
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='trace'}),flush=True)
    if not report['passed']:raise SystemExit('Replay failed')


if __name__=='__main__':main()
