"""Replay a complete100-batch capacity curriculum with weighted microbatches."""
import argparse,copy,json,math,random
import numpy as np
import torch
from train_conversation_v6 import ROOT,sha,load_rows,padded,text_batch
from spora.conversation_optimization import backward_supervised
from spora.growing import load
from spora.conversation import restore_conversation
from spora.language_training import interface_state,frozen_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',required=True);args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));path=ROOT/args.checkpoint;digest=sha(path)
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    payload=torch.load(path,map_location='cpu',weights_only=True);data=ROOT/plan['data'];curriculum=ROOT/plan['curriculum']
    if sha(data/'tokenizer.json')!=payload['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    stream=np.memmap(data/'pretrain-train.bin',dtype=np.uint16,mode='r');focus=json.loads((data/'focus-tokens.json').read_text('utf-8'))
    sources={key:load_rows(location,plan['length']) for key,location in {
        'dialogue':data/'dialogue-train-tokens.jsonl','teacher':curriculum/'teacher-tokens.jsonl','foundation':curriculum/'foundations-tokens.jsonl'}.items()}
    states=[];traces=[];preserved=[];torch.set_num_threads(2)
    from tokenizers import Tokenizer
    pad=Tokenizer.from_file(str(data/'tokenizer.json')).token_to_id('<|pad|>')
    for attempt in range(2):
        organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
        model=restore_conversation(organism,payload,plan['parent_sha256'],payload['tokenizer_sha256'],'cuda')
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95));opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        rng=random.Random();rng.setstate(payload['python_rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        cursor=payload['focus_cursor'];trace=[];batch_trace=payload['batch_trace_sha256']
        for step in range(payload['steps']+1,payload['steps']+101):
            fraction=min(step/plan['warmup'],1.) if step<=plan['warmup'] else .1+.9*.5*(1+math.cos(math.pi*(step-plan['warmup'])/(plan['schedule_steps']-plan['warmup'])))
            for group in opt.param_groups:group['lr']=plan['learning_rate']*fraction
            kind=plan['schedule'][(step-1)%len(plan['schedule'])]
            if kind=='pretrain':x,y=text_batch(stream,rng,plan['batch'],plan['length'],'cuda')
            else:
                rows=[rng.choice(sources[kind]) for _ in range(plan['batch'])]
                length=min(plan['length'],32*math.ceil((max(len(r['tokens']) for r in rows)-1)/32));x,y=padded(rows,length,pad,'cuda')
            if step%plan['focus_every']==0:
                seq=focus[cursor%len(focus)][:x.shape[1]+1];cursor+=1
                fx,fy=padded([dict(tokens=seq,mask=[True]*len(seq))],x.shape[1],pad,'cuda');x[0]=fx[0];y[0]=fy[0]
            import hashlib
            batch=hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
            batch_trace=hashlib.sha256(bytes.fromhex(batch_trace)+x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
            opt.zero_grad(set_to_none=True);loss=backward_supervised(model,state,x,y,plan['microbatch']);torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);opt.step()
            trace.append(dict(step=step,kind=kind,length=x.shape[1],batch_sha256=batch,loss=float(loss.detach()),batch_trace_sha256=batch_trace))
        traces.append(trace);states.append(interface_state(model));preserved.append(before==frozen_digest(organism,state));del opt,model,organism,state
    exact=all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    result=dict(checkpoint_sha256=digest,source_step=payload['steps'],traces_identical=traces[0]==traces[1],updated_tensors_exact=exact,
        old_individual_exact=all(preserved),checkpoint_unchanged=sha(path)==digest,trace=traces[0],
        scope='Two independent restores replay100 updates covering the100 scheduled batches and focus boundaries. Verifies saved-state repeatability, not a separate uninterrupted training trajectory.')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
    if not all(result[k] for k in ('traces_identical','updated_tensors_exact','old_individual_exact','checkpoint_unchanged')):raise SystemExit(1)

if __name__=='__main__':main()
