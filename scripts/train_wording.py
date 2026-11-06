"""Bounded P21 continuation, with source protection and unopened final tests."""
import argparse, json, time
from pathlib import Path
import torch
from torch.nn import functional as F
from wording_task import ROOT, TRAIN, DEVELOPMENT, batch, check, diagnostics
from ladder_task import tasks, sample, check as legacy_check, evidence_positions, matching_positions
from chinese_curriculum_common import plan_data, sha
from spora.growing import load
from spora.language_ladder import restore_ladder
from spora.language_bridge import LanguageBridge, causal_loss, PAD
from spora.language_training import load_interface, frozen_digest, interface_state, atomic_save
from spora.short_reasoning import response_loss
from spora.curriculum import CoveringSampler
from spora.language_eval import evaluate
from spora.text_data import windows

SOURCES = ('scripts/train_wording.py','scripts/wording_task.py','spora/language_ladder.py',
           'spora/language_bridge.py','scripts/ladder_task.py','scripts/reasoning_task.py',
           'spora/short_reasoning.py','spora/language_training.py')

def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def metrics(result):
    return [v for item in result.values() for k,v in item.items() if k!='records']

def protected_manifest():
    paths = list((ROOT/'checkpoints').glob('*.pt')) + list((ROOT/'checkpoints').glob('*.sqlite'))
    paths += list((ROOT/'runs').glob('p19*/**/*.pt')) + list((ROOT/'runs').glob('p20*/**/*.pt'))
    paths += [ROOT/'runs/chinese-curriculum-1000/checkpoint.pt']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(set(paths))}

def objective(model, state, tokens, starts, rows):
    logits = model(tokens,state)
    loss = 4*response_loss(logits,tokens,starts)
    # Scope guard: this phase has single-character, three-byte location labels.
    if any(len(r['answer'].encode('utf-8'))!=3 for r in rows):
        raise ValueError('Variable-length answers require a different alignment')
    device=tokens.device; offsets=torch.arange(3,device=device)
    b=torch.arange(len(rows),device=device)
    positions=torch.tensor(evidence_positions(rows),device=device)
    evidence=model.evidence_logits[b[:,None],starts[:,None]-1+offsets]
    loss=loss+F.cross_entropy(evidence.flatten(0,1),(positions[:,None]-2+offsets).flatten())
    pred_positions=torch.arange(1,tokens.shape[1],device=device)[None]
    mask=(pred_positions>=starts[:,None])&(tokens[:,1:]!=PAD)
    copy_labels=(pred_positions<starts[:,None]+3).float().expand_as(tokens[:,1:])
    loss=loss+F.binary_cross_entropy_with_logits(model.copy_gate_logits[:,:-1,0][mask],copy_labels[mask])
    content=model.content_logits[b[:,None],starts[:,None]-1+offsets]
    valid=torch.arange(tokens.shape[1],device=device)[None,None,:]<=starts[:,None,None]-1+offsets[None,:,None]
    content=content.masked_fill(~valid,torch.finfo(content.dtype).min)
    labels=torch.zeros_like(content)
    for index,matches in enumerate(matching_positions(rows)):
        for pos in matches: labels[index,offsets,pos-2+offsets]=1./len(matches)
    return loss-(labels*F.log_softmax(content,-1)).sum(-1).mean()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args(); plan=json.loads((ROOT/'configs/p21-wording.json').read_text('utf-8'))
    if args.seed not in plan['seeds']: raise ValueError('Unplanned seed')
    prior=json.loads((ROOT/'configs/p20j-content-matching.json').read_text('utf-8'))
    source_path=ROOT/f'runs/p20j-cortex-seed{args.seed}/checkpoint.pt'
    if sha(source_path)!=plan['source_sha256'][str(args.seed)]:raise ValueError('Source changed')
    if sha(prior['parent'])!=prior['parent_sha256']:raise ValueError('Parent changed')
    if sha(prior['source'])!=prior['source_sha256']:raise ValueError('Teacher changed')
    protected=protected_manifest()
    out=ROOT/f'runs/p21a-seed{args.seed}';out.mkdir(exist_ok=False)
    fingerprint={name:sha(name) for name in SOURCES}
    dump(out/'config.json',dict(plan=plan,source_code_sha256=fingerprint,protected=protected))
    torch.set_num_threads(2);torch.manual_seed(310000+args.seed)
    organism,state,_=load(ROOT/prior['parent'],'cuda');before=frozen_digest(organism,state)
    payload=torch.load(source_path,map_location='cpu',weights_only=True)
    model=restore_ladder(organism,payload,prior['parent_sha256'],'cuda')
    teacher=LanguageBridge(organism).cuda().eval()
    load_interface(teacher,torch.load(ROOT/prior['source'],map_location='cpu',weights_only=True),prior['parent_sha256'])
    _,docs,replay,groups,_=plan_data();replay=replay.cuda()
    sampler=CoveringSampler(groups,prior['replay_schedule'],311000+args.seed)
    with torch.no_grad():teacher_logits=torch.cat([teacher(replay[i:i+16],state) for i in range(0,len(replay),16)])
    dev=windows(docs,'development',128);data=tasks(expanded=True)
    small=sample(data['update']['development'],48,312000)
    baseline=dict(wording=check(model,state,small,DEVELOPMENT),diagnostics=diagnostics(model,state))
    dump(out/'baseline.json',baseline)
    print(json.dumps(dict(seed=args.seed,baseline=baseline['wording'],diagnostics={k:v['accuracy'] for k,v in baseline['diagnostics'].items()})),flush=True)
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01)
    generator=torch.Generator().manual_seed(313000+args.seed)
    started=time.perf_counter();logs=[];stable=0;status='budget_exhausted';targets=0;replay_targets=0
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,plan['steps_max']+1):
        rows=[];wordings=[]
        for i in range(16):
            stage=('single','select','update')[i%3] if i<4 else 'update'
            values=data[stage]['train'];rows.append(values[int(torch.randint(len(values),(1,),generator=generator))])
            wordings.append((0,int(torch.randint(2,(1,),generator=generator))) if i<4 else TRAIN[int(torch.randint(len(TRAIN),(1,),generator=generator))])
        tokens,starts=batch(rows,wordings);tokens=tokens.cuda();starts=starts.cuda()
        optimizer.zero_grad(set_to_none=True);loss=objective(model,state,tokens,starts,rows);loss.backward()
        ix=sampler.take(8);old=replay[ix];logits=model(old,state);valid=old[:,1:]!=PAD
        replay_loss=causal_loss(logits,old)
        kl=F.kl_div(F.log_softmax(logits[:,:-1][valid],-1),F.softmax(teacher_logits[ix,:-1][valid],-1),reduction='batchmean')
        (.5*replay_loss+.5*kl).backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
        targets+=int(((torch.arange(tokens.shape[1],device='cuda')[None]>=starts[:,None])&(tokens!=PAD)).sum())
        replay_targets+=int(valid.sum())
        if step%plan['check_every']==0:
            development=check(model,state,small,DEVELOPMENT)
            legacy={s:legacy_check(model,state,sample(d['development'],48,312001),s) for s,d in data.items()}
            nll=evaluate(model,state,dev)['negative_log_likelihood']
            passed=min(metrics(development)+metrics(legacy))>=plan['gate'] and nll<=prior['source_development_nll']*prior['retention_ratio_max']
            stable=stable+1 if passed else 0;full=None
            if stable>=2:
                full=dict(wording=check(model,state,data['update']['development'],DEVELOPMENT),
                    legacy={s:legacy_check(model,state,d['development'],s) for s,d in data.items()})
                if min(metrics(full['wording'])+metrics(full['legacy']))<plan['gate']:stable=0
            if frozen_digest(organism,state)!=before:raise AssertionError('Old individual changed')
            row=dict(step=step,loss=float(loss.detach()),replay_loss=float(replay_loss.detach()),kl=float(kl.detach()),
                     development=development,legacy=legacy,nll=nll,stable=stable,full=full)
            logs.append(row);dump(out/'development.json',logs);print(json.dumps(dict(seed=args.seed,**row)),flush=True)
            snapshot={k:v for k,v in payload.items() if k not in ('state','optimizer','sampler','task_rng','torch_rng','cuda_rng')}
            snapshot.update(state=interface_state(model),optimizer=optimizer.state_dict(),sampler=sampler.snapshot(),
                task_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                phase='P21a',continuation_source_sha256=sha(source_path),continuation_steps=step,
                continuation_plan_sha256=sha('configs/p21-wording.json'),task_targets=targets,replay_targets=replay_targets)
            atomic_save(snapshot,out/'checkpoint.pt')
            if stable>=2:status='development_passed';break
    if protected_manifest()!=protected:raise AssertionError('Protected files changed')
    if fingerprint!={name:sha(name) for name in SOURCES}:raise AssertionError('Implementation changed during training')
    result=dict(seed=args.seed,status=status,steps=step,task_targets=targets,replay_targets=replay_targets,
        seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        old_tensors_exact=before==frozen_digest(organism,state),replay_coverage=sampler.counts,
        checkpoint_sha256=sha(out/'checkpoint.pt'),checks=logs)
    dump(out/'training.json',result)
    dump(out/'diagnostics.json',diagnostics(model,state))
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','replay_coverage')}),flush=True)

if __name__=='__main__':main()
