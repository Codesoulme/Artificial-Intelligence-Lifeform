"""P23c continuation with whole-source curriculum; original random lineage preserved."""
import argparse,json,math,random,time,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from dataclasses import asdict
from spora.conversation import ConversationBridge,ConversationConfig,restore_conversation,generate
from spora.conversation_data import DialogueRows
from spora.growing import load
from spora.language_training import interface_state,atomic_save,frozen_digest
from chinese_curriculum_common import memory_answers
from types import SimpleNamespace
from prepare_conversation_v2 import sha,SPECIAL
from train_wording import dump,protected_manifest
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('spora/conversation.py','spora/conversation_data.py','scripts/train_conversation_v3.py','scripts/prepare_conversation_v4.py','scripts/prepare_conversation_v2.py','scripts/curate_conversation.py','spora/language_training.py')
PROMPTS=('你好，请简单介绍你的能力。','我今天有点累，可以怎样放松？','请用一句话解释为什么会下雨。','把“明天我去图书馆看书”改成疑问句。','我不知道该选哪本书，你能先问我一个问题吗？','社会主义核心价值观的个人层面包括哪些内容？')

def load_rows(path,length):
    return DialogueRows(path,length)

def padded(rows,length,pad,device):
    tokens=[];labels=[]
    for row in rows:
        seq=row['tokens'];mask=row['mask']
        padded_seq=seq+[pad]*(length+1-len(seq))
        target=[t if valid else -100 for t,valid in zip(seq[1:],mask[1:])]+[-100]*(length+1-len(seq))
        tokens.append(padded_seq[:-1]);labels.append(target)
    return torch.tensor(tokens,device=device),torch.tensor(labels,device=device)

def text_batch(stream,rng,batch,length,device):
    starts=[rng.randrange(len(stream)-length) for _ in range(batch)]
    seq=np.stack([np.array(stream[i:i+length+1],dtype=np.int64) for i in starts])
    return torch.tensor(seq[:,:-1],device=device),torch.tensor(seq[:,1:],device=device)

def batch_loss(model,state,x,y):
    return F.cross_entropy(model(x,state).flatten(0,1),y.flatten(),ignore_index=-100)

@torch.no_grad()
def monitor(model,state,tokenizer,development,dialogue_dev,length,pad):
    rng=random.Random(230101);device=next(model.parameters()).device
    total=0.;count=0
    for _ in range(8):
        x,y=text_batch(development,rng,4,length,device);loss=batch_loss(model,state,x,y)
        total+=float(loss)*y.numel();count+=y.numel()
    sft_total=0.;sft_count=0
    for start in range(0,min(32,len(dialogue_dev)),4):
        x,y=padded(dialogue_dev[start:start+4],length,pad,device);n=int((y!=-100).sum())
        sft_total+=float(batch_loss(model,state,x,y))*n;sft_count+=n
    generated=[];eos=tokenizer.token_to_id('<|eos|>');bos=tokenizer.token_to_id('<|bos|>')
    user=tokenizer.token_to_id('<|user|>');assistant=tokenizer.token_to_id('<|assistant|>')
    for prompt in PROMPTS:
        prefix=[bos,user]+tokenizer.encode(prompt).ids+[eos,assistant]
        result=generate(model,state,prefix,eos,budget=96,forbidden=(pad,bos,user,assistant))
        text=tokenizer.decode(result['tokens'],skip_special_tokens=False)
        result.update(prompt=prompt,text=text,utf8_valid='\ufffd' not in text)
        generated.append(result)
    return dict(pretrain_nll=total/count,dialogue_nll=sft_total/sft_count,samples=generated,
                scope='Fixed development samples, not a chat acceptance test. NLL only comparable within this tokenizer.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='configs/p23c-conversation.json')
    parser.add_argument('--output',default='runs/p23c-conversation');parser.add_argument('--until',type=int,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();plan=json.loads((ROOT/args.plan).read_text('utf-8'));data=ROOT/plan['data'];out=ROOT/args.output
    if not 0<args.until<=plan['schedule_steps'] or args.until%plan['check_every']:raise ValueError('End at a planned checkpoint')
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual file changed')
    manifest=json.loads((data/'manifest.json').read_text('utf-8'))
    audit_paths=plan['data_audits'];audits={}
    for name in audit_paths:
        report=json.loads((ROOT/name).read_text('utf-8'))
        if not report['passed']:raise ValueError('Data isolation audit failed: '+name)
        for artifact,digest in report.get('data_artifact_sha256',{}).items():
            if manifest['files'][artifact]!=digest:raise ValueError('Audit covers different data')
        lineage=report.get('manifest_sha256')
        if lineage is not None:lineage={key.replace('\\','/'):value for key,value in lineage.items()}
        if lineage is not None and lineage[plan['data']]!=sha(data/'manifest.json'):raise ValueError('Lineage audit is stale')
        audits[name]=sha(ROOT/name)
    # Verify every local input used for training and monitoring, not final test.
    inputs=('tokenizer.json','pretrain-train.bin','pretrain-development.bin','dialogue-train-tokens.jsonl','dialogue-development-tokens.jsonl','focus-tokens.json')
    for name in inputs:
        if sha(data/name)!=manifest['files'][name]:raise ValueError('Dataset artifact changed: '+name)
    tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'));pad=tokenizer.token_to_id('<|pad|>')
    protected=protected_manifest()
    for pattern in ('p21*/**/*.pt','p22*/**/*.pt','p23a*/**/*.pt','p23b*/**/*.pt'):
        for path in (ROOT/'runs').glob(pattern):protected[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    fingerprints={name:sha(ROOT/name) for name in SOURCES}
    metadata=dict(plan=plan,plan_sha256=sha(ROOT/args.plan),manifest_sha256=sha(data/'manifest.json'),data_audit_sha256=audits,source_code_sha256=fingerprints,protected=protected)
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);rng=random.Random(plan['seed'])
    organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
    payload=None;step=0;targets=0;focus_cursor=0;counts={};logs=[];elapsed=0.
    if args.resume:
        if json.loads((out/'config.json').read_text('utf-8'))!=metadata:raise ValueError('Resume requires exact implementation/data/plan/protected sources')
        payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
        model=restore_conversation(organism,payload,plan['parent_sha256'],manifest['tokenizer_sha256'],'cuda')
        step=payload['steps'];targets=payload['targets'];focus_cursor=payload['focus_cursor'];counts=payload['counts'];elapsed=payload['seconds']
        if args.until<=step:raise ValueError('Resume must make progress')
        rng.setstate(payload['python_rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        logs=json.loads((out/'development.json').read_text('utf-8'))
    else:
        out.mkdir(exist_ok=False);dump(out/'config.json',metadata)
        initial=ROOT/plan['initial_language']
        if sha(initial)!=plan['initial_language_sha256']:raise ValueError('Language parent changed')
        language_payload=torch.load(initial,map_location='cpu',weights_only=True)
        model=restore_conversation(organism,language_payload,plan['parent_sha256'],manifest['tokenizer_sha256'],'cuda')
        if asdict(model.config)!={**plan['model'],'memory_layers':tuple(plan['model']['memory_layers'])}:raise ValueError('Architecture differs from language parent')
        # Explicit curriculum fork: retain learned tensors, reset optimizer and
        # schedule, keep ancestor steps/targets in plan (not silently resuming).
        if language_payload['steps']!=plan['ancestor_steps'] or language_payload['targets']!=plan['ancestor_targets']:raise ValueError('Ancestor accounting differs')
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
    if payload is not None:optimizer.load_state_dict(payload['optimizer'])
    stream=np.memmap(data/'pretrain-train.bin',dtype=np.uint16,mode='r');dev=np.memmap(data/'pretrain-development.bin',dtype=np.uint16,mode='r')
    sft=load_rows(data/'dialogue-train-tokens.jsonl',plan['length']);sft_dev=load_rows(data/'dialogue-development-tokens.jsonl',plan['length'])
    focus=json.loads((data/'focus-tokens.json').read_text('utf-8'))
    if not sft or not sft_dev or not focus:raise ValueError('Missing curriculum')
    if payload is None:
        baseline=monitor(model,state,tokenizer,dev,sft_dev,plan['length'],pad);dump(out/'baseline.json',baseline)
        atomic_save(dict(format='spora-conversation-v1',organism_sha256=plan['parent_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],config=asdict(model.config),state=interface_state(model)),out/'language-initial.pt')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(start_step=step,until=args.until,parameters=sum(p.numel() for p in model.trainable_parameters()),sft_rows=len(sft),focus_segments=len(focus))),flush=True)
    for step in range(step+1,args.until+1):
        fraction=min(step/plan['warmup'],1.) if step<=plan['warmup'] else .1+.9*.5*(1+math.cos(math.pi*(step-plan['warmup'])/(plan['schedule_steps']-plan['warmup'])))
        for group in optimizer.param_groups:group['lr']=plan['learning_rate']*fraction
        kind=plan['schedule'][(step-1)%len(plan['schedule'])]
        if kind=='pretrain':x,y=text_batch(stream,rng,plan['batch'],plan['length'],'cuda')
        else:x,y=padded([rng.choice(sft) for _ in range(plan['batch'])],plan['length'],pad,'cuda')
        if step%plan['focus_every']==0:
            seq=focus[focus_cursor%len(focus)][:plan['length']+1];focus_cursor+=1
            fx,fy=padded([dict(tokens=seq,mask=[True]*len(seq))],plan['length'],pad,'cuda')
            x[0]=fx[0];y[0]=fy[0];counts['focus_targets']=counts.get('focus_targets',0)+int((fy!=-100).sum())
        optimizer.zero_grad(set_to_none=True);loss=batch_loss(model,state,x,y)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
        n=int((y!=-100).sum());targets+=n;counts[kind+'_steps']=counts.get(kind+'_steps',0)+1
        if step%50==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),targets=targets,seconds=elapsed+time.perf_counter()-start)),flush=True)
        if step%plan['check_every']==0:
            report=monitor(model,state,tokenizer,dev,sft_dev,plan['length'],pad)
            if before!=frozen_digest(organism,state):raise AssertionError('Original individual changed')
            row=dict(step=step,loss=float(loss.detach()),targets=targets,**report);logs.append(row)
            snapshot=dict(format='spora-conversation-v1',organism_sha256=plan['parent_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],
                config=asdict(model.config),state=interface_state(model),optimizer=optimizer.state_dict(),steps=step,targets=targets,
                focus_cursor=focus_cursor,counts=counts,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                seconds=elapsed+time.perf_counter()-start,plan_sha256=metadata['plan_sha256'],manifest_sha256=metadata['manifest_sha256'])
            atomic_save(snapshot,out/'checkpoint.pt')
            if step%plan['snapshot_every']==0:atomic_save(snapshot,out/f'step{step:06d}.pt')
            dump(out/'development.json',logs)
            print(json.dumps(dict(step=step,pretrain_nll=report['pretrain_nll'],dialogue_nll=report['dialogue_nll'],samples=[dict(prompt=r['prompt'],characters=len(r['text']),ended=r['ended_with_eos'],utf8_valid=r['utf8_valid']) for r in report['samples']])),flush=True)
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Protected artifact changed: '+name)
    if fingerprints!={name:sha(ROOT/name) for name in SOURCES}:raise AssertionError('Implementation changed during segment')
    result=dict(status='segment_completed_not_chat_accepted',steps=step,targets=targets,counts=counts,focus_cursor=focus_cursor,
        old_tensors_exact=before==frozen_digest(organism,state),memory=memory_answers(SimpleNamespace(model=model,state=state)),
        seconds=elapsed+time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(out/'checkpoint.pt'))
    dump(out/f'segment-{step:06d}.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
