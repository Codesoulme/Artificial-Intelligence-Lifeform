"""Read-only one-update replay from a stored optimizer/RNG checkpoint."""
import argparse,json,random,copy,math
import numpy as np
import torch
from train_conversation import ROOT,sha,dump,text_batch,batch_loss
from spora.growing import load
from spora.conversation import restore_conversation
from spora.language_training import interface_state,frozen_digest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='configs/p23-conversation.json')
    parser.add_argument('--checkpoint',default='runs/p23a-conversation/checkpoint.pt')
    parser.add_argument('--output',default='runs/p23a-conversation/resume-verification.json')
    args=parser.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    path=ROOT/args.checkpoint;before_file=sha(path)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));payload=torch.load(path,map_location='cpu',weights_only=True)
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    if sha(ROOT/plan['data']/'tokenizer.json')!=payload['tokenizer_sha256']:raise ValueError('Vocabulary changed')
    torch.set_num_threads(2);stream=np.memmap(ROOT/plan['data']/'pretrain-train.bin',mode='r',dtype=np.uint16)
    states=[];losses=[];inputs=[];protected=[]
    for _ in range(2):
        organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
        model=restore_conversation(organism,payload,plan['parent_sha256'],payload['tokenizer_sha256'],'cuda')
        opt=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95));opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        rng=random.Random();rng.setstate(payload['python_rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        step=payload['steps']+1
        if plan['schedule'][(step-1)%len(plan['schedule'])]!='pretrain' or step%plan['focus_every']==0:raise ValueError('Verification expects next ordinary pretraining step')
        fraction=.1+.9*.5*(1+math.cos(math.pi*(step-plan['warmup'])/(plan['schedule_steps']-plan['warmup'])))
        for group in opt.param_groups:group['lr']=plan['learning_rate']*fraction
        x,y=text_batch(stream,rng,plan['batch'],plan['length'],'cuda');inputs.append(x.cpu())
        opt.zero_grad(set_to_none=True);loss=batch_loss(model,state,x,y);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);opt.step()
        states.append(interface_state(model));losses.append(float(loss.detach()));protected.append(before==frozen_digest(organism,state))
        del opt,model,organism,state
    exact=all(torch.equal(t,states[1][k]) for k,t in states[0].items())
    result=dict(checkpoint_sha256=before_file,source_step=payload['steps'],sample_identical=torch.equal(*inputs),losses=losses,
        replayed_update_exact=exact,maximum_tensor_difference=max(float((t-states[1][k]).abs().max()) for k,t in states[0].items()),
        old_individual_exact=all(protected),checkpoint_unchanged=sha(path)==before_file,
        scope='Two independent restores replay one identical update; validates saved optimizer/RNG replay. Does not compare against a separately uninterrupted long training trajectory.')
    dump(out,result);print(json.dumps(result),flush=True)
    if not all(result[k] for k in ('sample_identical','replayed_update_exact','old_individual_exact','checkpoint_unchanged')):raise SystemExit(1)

if __name__=='__main__':main()
