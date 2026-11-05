"""Adaptive bounded development ladder. Every failed curriculum is retained."""
import argparse,json,time
import torch
from torch.nn import functional as F
from ladder_task import ROOT,tasks,sample,check,batch,changed_query,evidence_positions,matching_positions
from chinese_curriculum_common import plan_data
from spora.curriculum import CoveringSampler
from spora.growing import load
from spora.language_bridge import LanguageBridge,causal_loss,PAD
from spora.language_ladder import CortexLanguageBridge
from spora.short_reasoning import response_loss
from spora.language_training import load_interface,frozen_digest,interface_state,atomic_save
from spora.language_eval import evaluate
from spora.text_data import windows
from audit_short_reasoning import sha,protected_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['original','cortex'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--plan',default='configs/p20-language-ladder.json');p.add_argument('--tag',default='p20')
    args=p.parse_args();plan=json.loads((ROOT/args.plan).read_text('utf-8'))
    if not args.tag.replace('-','').isalnum():raise ValueError('Simple run tag required')
    if args.seed not in plan['seeds']:raise ValueError('Unplanned seed')
    protected_files(plan)
    out=ROOT/f'runs/{args.tag}-{args.arm}-seed{args.seed}';out.mkdir(exist_ok=False)
    (out/'config.json').write_text(json.dumps({'plan':plan,'arm':args.arm,'seed':args.seed,
        'source_code_sha256':{name:sha(name) for name in ('scripts/train_language_ladder.py','scripts/ladder_task.py','spora/language_ladder.py','spora/language_bridge.py')}},indent=2),encoding='utf-8')
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
    source=torch.load(ROOT/plan['source'],map_location='cpu',weights_only=True)
    teacher=LanguageBridge(organism).cuda().eval();load_interface(teacher,source,plan['parent_sha256'])
    if args.arm=='cortex':model=CortexLanguageBridge(organism,source,plan['parent_sha256'],depth=plan.get('depth',2),local_bytes=plan.get('local_bytes',False),evidence_reader=plan.get('evidence_reader',False),copy_reader=plan.get('copy_reader',False),local_evidence=plan.get('local_evidence',False),query_window=plan.get('query_window',32)).cuda()
    else:model=LanguageBridge(organism).cuda();load_interface(model,source,plan['parent_sha256'])
    _,docs,replay,groups,_=plan_data();sampler=CoveringSampler(groups,plan['replay_schedule'],2940000+args.seed)
    replay=replay.cuda();dev=windows(docs,'development',128)
    with torch.no_grad():teacher_logits=torch.cat([teacher(replay[i:i+16],state) for i in range(0,len(replay),16)])
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01)
    data=tasks(expanded=plan.get('expanded_objects',False));generator=torch.Generator().manual_seed(2950000+args.seed);logs=[];total=0;targets=0;replay_targets=0;started=time.perf_counter()
    stages_done=[];status='budget_exhausted'
    torch.cuda.reset_peak_memory_stats()
    for stage,limit in plan['stages'].items():
        stable=0
        for step in range(1,limit+1):
            rows=[]
            for i in range(plan['task_batch']):
                if plan.get('paired_batches',False) and i%2==1 and len(rows[-1]['pair'])==2:
                    rows.append(changed_query(rows[-1]));continue
                source_stage=stage
                if i<3 and stages_done:source_stage=stages_done[int(torch.randint(len(stages_done),(1,),generator=generator))]
                values=data[source_stage]['train'];rows.append(values[int(torch.randint(len(values),(1,),generator=generator))])
            templates=torch.randint(2,(len(rows),),generator=generator).tolist();tokens,starts=batch(rows,'answer',templates)
            tokens=tokens.cuda();starts=starts.cuda();ix=sampler.take(plan['replay_batch']);old=replay[ix]
            optimizer.zero_grad(set_to_none=True)
            task_loss=response_loss(model(tokens,state),tokens,starts)
            objective=plan.get('task_weight',1.)*task_loss
            evidence_loss=None
            if plan.get('evidence_reader',False):
                positions=torch.tensor(evidence_positions(rows),device='cuda')
                batch_indices=torch.arange(len(rows),device='cuda')
                if plan.get('copy_reader',False):
                    # Every task answer begins with one three-byte location.
                    # This alignment is a training label, never an inference rule.
                    offsets=torch.arange(3,device='cuda')
                    evidence_loss=F.cross_entropy(model.evidence_logits[batch_indices[:,None],starts[:,None]-1+offsets].flatten(0,1),
                        (positions[:,None]-2+offsets).flatten())
                    pred_positions=torch.arange(tokens.shape[1]-1,device='cuda')[None]+1
                    mask=(pred_positions>=starts[:,None])&(tokens[:,1:]!=PAD)
                    copy_labels=(pred_positions<starts[:,None]+3).float().expand_as(tokens[:,1:])
                    gate_loss=F.binary_cross_entropy_with_logits(model.copy_gate_logits[:,:-1,0][mask],copy_labels[mask])
                    objective=objective+plan.get('gate_weight',1.)*gate_loss
                    if plan.get('matching_weight',0.):
                        if not plan.get('local_evidence',False):raise ValueError('Matching supervision requires content logits')
                        content=model.content_logits[batch_indices[:,None],starts[:,None]-1+offsets]
                        valid_keys=torch.arange(tokens.shape[1],device='cuda')[None,None,:] <= (starts[:,None,None]-1+offsets[None,:,None])
                        content=content.masked_fill(~valid_keys,torch.finfo(content.dtype).min)
                        labels=torch.zeros_like(content)
                        for b,matches in enumerate(matching_positions(rows)):
                            for pos in matches:
                                labels[b,offsets,pos-2+offsets]=1./len(matches)
                        matching_loss=-(labels*F.log_softmax(content,-1)).sum(-1).mean()
                        objective=objective+plan['matching_weight']*matching_loss
                else:evidence_loss=F.cross_entropy(model.evidence_logits[batch_indices,starts-1],positions)
                objective=objective+plan.get('evidence_weight',1.)*evidence_loss
            objective.backward()
            logits=model(old,state);replay_loss=causal_loss(logits,old)
            valid=old[:,1:]!=PAD
            kl=F.kl_div(F.log_softmax(logits[:,:-1][valid],-1),F.softmax(teacher_logits[ix,:-1][valid],-1),reduction='batchmean')
            (plan['replay_weight']*replay_loss+plan['distillation_weight']*kl).backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=1;targets+=int(((torch.arange(tokens.shape[1],device='cuda')[None]>=starts[:,None])&(tokens!=PAD)).sum());replay_targets+=int(valid.sum())
            if step%plan['check_every']==0 or step==limit:
                fit=check(model,state,sample(data[stage]['train'],48,plan['development_seed']),stage)
                development=check(model,state,sample(data[stage]['development'],48,plan['development_seed']),stage)
                previous={s:check(model,state,sample(data[s]['development'],48,plan['development_seed']),s) for s in stages_done}
                retention=evaluate(model,state,dev)['negative_log_likelihood']
                passed=min(list(fit.values())+list(development.values())+[v for r in previous.values() for v in r.values()])>=.95 and retention<=plan['source_development_nll']*plan['retention_ratio_max']
                stable=stable+1 if passed else 0
                full_development=None
                if stable>=2 and plan.get('full_development_gate',False):
                    full_development={s:check(model,state,data[s]['development'],s) for s in stages_done+[stage]}
                    if min(v for r in full_development.values() for v in r.values())<.95:stable=0
                if frozen_digest(organism,state)!=before:raise AssertionError('Old neural state changed')
                row=dict(stage=stage,step=step,total=total,task_loss=float(task_loss.detach()),replay_loss=float(replay_loss.detach()),kl=float(kl.detach()),fit=fit,development=development,previous=previous,language_nll=retention,stable=stable)
                if full_development is not None:row['full_development']=full_development
                if evidence_loss is not None:row['evidence_loss']=float(evidence_loss.detach())
                logs.append(row);print(json.dumps(dict(arm=args.arm,seed=args.seed,**row)),flush=True)
                snapshot={'format':'spora-language-ladder-v1','arm':args.arm,'seed':args.seed,'organism_sha256':plan['parent_sha256'],
                    'source_sha256':plan['source_sha256'],'plan_sha256':sha(args.plan),'depth':plan.get('depth',2),'local_bytes':plan.get('local_bytes',False),'evidence_reader':plan.get('evidence_reader',False),'copy_reader':plan.get('copy_reader',False),'local_evidence':plan.get('local_evidence',False),'query_window':plan.get('query_window',32),'state':interface_state(model),
                    'stage':stage,'stage_step':step,'steps':total,'optimizer':optimizer.state_dict(),'sampler':sampler.snapshot(),
                    'task_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
                atomic_save(snapshot,out/'checkpoint.pt')
                (out/'development.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
                if stable>=2:
                    atomic_save(snapshot,out/f'{stage}-passed.pt')
                    break
        if stable<2:break
        stages_done.append(stage)
    if len(stages_done)==len(plan['stages']):status='development_passed'
    protected_files(plan)
    result=dict(arm=args.arm,seed=args.seed,status=status,stages_done=stages_done,steps=total,
        trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),task_targets=targets,replay_targets=replay_targets,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        old_tensors_exact=frozen_digest(organism,state)==before,replay_coverage=sampler.counts,
        checkpoint_sha256=sha(out/'checkpoint.pt'),checks=logs)
    (out/'training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','replay_coverage')}),flush=True)


if __name__=='__main__':main()
