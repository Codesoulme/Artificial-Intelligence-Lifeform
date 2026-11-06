"""P22 fixed-budget controlled continuation. No final-test generation here."""
import argparse,json,random,time,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from query_task import ROOT,TRAIN,DEV,wording,unknown,batch,check,evaluation_rows,gate_values
from scripts.wording_task import TRAIN as SHORT_TRAIN,check as short_check,DEVELOPMENT as SHORT_DEV
from scripts.ladder_task import tasks,sample,check as legacy_check,evidence_positions,matching_positions
from scripts.train_wording import dump,protected_manifest,metrics
from scripts.chinese_curriculum_common import plan_data,sha,memory_answers
from spora.growing import load
from spora.language_ladder import restore_ladder
from spora.query_bridge import extend_query,restore_query
from spora.language_bridge import LanguageBridge,causal_loss,PAD
from spora.language_training import load_interface,frozen_digest,interface_state,atomic_save
from spora.short_reasoning import response_loss
from spora.curriculum import CoveringSampler
from spora.language_eval import evaluate
from spora.text_data import windows

SOURCES=('scripts/train_query.py','scripts/query_task.py','spora/query_bridge.py',
         'spora/language_ladder.py','scripts/wording_task.py','scripts/ladder_task.py',
         'spora/language_bridge.py','spora/short_reasoning.py','spora/language_training.py')

def protection():
    result=protected_manifest()
    for p in (ROOT/'runs').glob('p21*/**/*.pt'):result[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    return result

def objective(model,state,tokens,starts,rows):
    logits=model(tokens,state);loss=4*response_loss(logits,tokens,starts)
    known=torch.tensor([bool(r['indices']) for r in rows],device=tokens.device)
    positions=torch.arange(1,tokens.shape[1],device=tokens.device)[None]
    response=(positions>=starts[:,None])&(tokens[:,1:]!=PAD)
    copy=((positions<starts[:,None]+3)&known[:,None]).float()
    loss=loss+F.binary_cross_entropy_with_logits(model.copy_gate_logits[:,:-1,0][response],copy[response])
    if bool(known.any()):
        selected=known.nonzero().flatten();subset=[r for r in rows if r['indices']]
        if any(len(r['answer'].encode('utf-8'))!=3 for r in subset):raise ValueError('Single-character known locations required')
        offsets=torch.arange(3,device=tokens.device);starts_known=starts[selected]
        target=torch.tensor(evidence_positions(subset),device=tokens.device)
        evidence=model.evidence_logits[selected[:,None],starts_known[:,None]-1+offsets]
        loss=loss+F.cross_entropy(evidence.flatten(0,1),(target[:,None]-2+offsets).flatten())
        content=model.content_logits[selected[:,None],starts_known[:,None]-1+offsets]
        causal=torch.arange(tokens.shape[1],device=tokens.device)[None,None,:]<=starts_known[:,None,None]-1+offsets[None,:,None]
        content=content.masked_fill(~causal,torch.finfo(content.dtype).min);labels=torch.zeros_like(content)
        for i,matches in enumerate(matching_positions(subset)):
            for pos in matches:labels[i,offsets,pos-2+offsets]=1./len(matches)
        loss=loss-(labels*F.log_softmax(content,-1)).sum(-1).mean()
    return loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['local','context'],required=True)
    p.add_argument('--plan',default='configs/p22-query.json');p.add_argument('--tag',default='p22a')
    args=p.parse_args();plan=json.loads((ROOT/args.plan).read_text('utf-8'))
    if args.seed not in plan['seeds'] or not args.tag.replace('-','').isalnum():raise ValueError('Invalid seed/tag')
    prior=json.loads((ROOT/'configs/p20j-content-matching.json').read_text('utf-8'))
    source=ROOT/f'runs/p21a-seed{args.seed}/checkpoint.pt'
    if sha(source)!=plan['source_sha256'][str(args.seed)]:raise ValueError('Source differs')
    if sha(prior['parent'])!=prior['parent_sha256'] or sha(prior['source'])!=prior['source_sha256']:raise ValueError('Parent/teacher changed')
    protected=protection();fingerprint={name:sha(name) for name in SOURCES}
    out=ROOT/f'runs/{args.tag}-{args.arm}-seed{args.seed}';out.mkdir(exist_ok=False)
    dump(out/'config.json',dict(plan=plan,source_code_sha256=fingerprint,protected=protected))
    torch.set_num_threads(2);torch.manual_seed(320000+args.seed)
    organism,state,_=load(ROOT/prior['parent'],'cuda');before=frozen_digest(organism,state)
    payload=torch.load(source,map_location='cpu',weights_only=True)
    model=(extend_query if args.arm=='context' else restore_ladder)(organism,payload,prior['parent_sha256'],'cuda')
    teacher=LanguageBridge(organism).cuda().eval()
    load_interface(teacher,torch.load(ROOT/prior['source'],map_location='cpu',weights_only=True),prior['parent_sha256'])
    _,docs,replay,groups,_=plan_data();replay=replay.cuda()
    sampler=CoveringSampler(groups,prior['replay_schedule'],321000+args.seed)
    with torch.no_grad():teacher_logits=torch.cat([teacher(replay[i:i+16],state) for i in range(0,len(replay),16)])
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01)
    rng=random.Random(320100+args.seed);data=tasks(expanded=True);known,absent=evaluation_rows('development',48)
    dev=windows(docs,'development',128);templates=[wording(t) for t in DEV]+[(0,0)]
    baseline=check(model,state,known,absent,templates)
    dump(out/'baseline.json',baseline);print(json.dumps(dict(seed=args.seed,arm=args.arm,baseline=baseline)),flush=True)
    logs=[];stable=0;targets=0;replay_targets=0;started=time.perf_counter();schedule=hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,plan['steps']+1):
        rows=[];wordings=[]
        for i in range(16):
            if i<4:
                stage=('single','select','update')[i%3];row=rng.choice(data[stage]['train']);template=rng.choice(SHORT_TRAIN)
                if step%2==0 and i<2:row=unknown(row,rng)
            else:
                row=rng.choice(data['update']['train']);template=wording(rng.choice(TRAIN))
                if i>=12:row=unknown(row,rng)
            rows.append(row);wordings.append(template)
        tokens,starts=batch(rows,wordings);schedule.update(tokens.numpy().tobytes());tokens=tokens.cuda();starts=starts.cuda()
        optimizer.zero_grad(set_to_none=True);loss=objective(model,state,tokens,starts,rows);loss.backward()
        ix=sampler.take(8);old=replay[ix];logits=model(old,state);valid=old[:,1:]!=PAD
        replay_loss=causal_loss(logits,old);kl=F.kl_div(F.log_softmax(logits[:,:-1][valid],-1),F.softmax(teacher_logits[ix,:-1][valid],-1),reduction='batchmean')
        (.5*replay_loss+.5*kl).backward();torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
        targets+=int(((torch.arange(tokens.shape[1],device='cuda')[None]>=starts[:,None])&(tokens!=PAD)).sum());replay_targets+=int(valid.sum())
        if step%plan['check_every']==0:
            development=check(model,state,known,absent,templates)
            legacy={s:legacy_check(model,state,sample(d['development'],48,322010),s) for s,d in data.items()}
            short=short_check(model,state,known,SHORT_DEV);nll=evaluate(model,state,dev)['negative_log_likelihood']
            passed=min(gate_values(development)+metrics(legacy)+metrics(short))>=plan['gate'] and nll<=prior['source_development_nll']*prior['retention_ratio_max']
            stable=stable+1 if passed else 0
            if before!=frozen_digest(organism,state):raise AssertionError('Old individual changed')
            row=dict(step=step,loss=float(loss.detach()),replay_loss=float(replay_loss.detach()),kl=float(kl.detach()),development=development,legacy=legacy,short=short,nll=nll,stable=stable)
            logs.append(row);dump(out/'development.json',logs);print(json.dumps(dict(seed=args.seed,arm=args.arm,**row)),flush=True)
            snapshot=dict(format='spora-context-query-v1',arm=args.arm,seed=args.seed,organism_sha256=prior['parent_sha256'],
                source_sha256=sha(source),plan_sha256=sha(args.plan),state=interface_state(model),optimizer=optimizer.state_dict(),
                sampler=sampler.snapshot(),python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                steps=step,task_targets=targets,replay_targets=replay_targets,schedule_sha256=schedule.hexdigest())
            atomic_save(snapshot,out/'checkpoint.pt')
    full=None;status='budget_exhausted'
    if stable>=2:
        fk,fu=evaluation_rows('development',864)
        full=dict(development=check(model,state,fk,fu,templates),legacy={s:legacy_check(model,state,d['development'],s) for s,d in data.items()},short=short_check(model,state,fk,SHORT_DEV))
        if min(gate_values(full['development'])+metrics(full['legacy'])+metrics(full['short']))>=plan['gate']:status='development_passed'
    if protection()!=protected or before!=frozen_digest(organism,state):raise AssertionError('Protected data changed')
    if fingerprint!={name:sha(name) for name in SOURCES}:raise AssertionError('Implementation changed')
    from types import SimpleNamespace
    result=dict(seed=args.seed,arm=args.arm,status=status,steps=step,task_targets=targets,replay_targets=replay_targets,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),old_tensors_exact=True,
        memory=memory_answers(SimpleNamespace(model=model,state=state)),replay_coverage=sampler.counts,
        checkpoint_sha256=sha(out/'checkpoint.pt'),schedule_sha256=schedule.hexdigest(),checks=logs,full=full)
    dump(out/'training.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','replay_coverage','full')}),flush=True)

if __name__=='__main__':main()
