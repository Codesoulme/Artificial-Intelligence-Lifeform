"""Fixed P19 paired trials; only external task labels/replay drive updates."""
import argparse,hashlib,json,time
import torch
from reasoning_task import ROOT,dataset,balanced,batch,audit
from spora.growing import load
from spora.language_bridge import LanguageBridge,causal_loss,PAD
from spora.language_training import load_interface,interface_state,atomic_save,frozen_digest
from spora.short_reasoning import RecurrentLanguageBridge,response_loss
from spora.text_data import read_manifest,windows


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--start-training',action='store_true');args=parser.parse_args()
    if not args.start_training:parser.error('Explicit start required')
    plan_path=ROOT/'configs/p19-short-reasoning.json';plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if args.arm not in plan['arms'] or args.seed not in plan['seeds']:raise ValueError('Unplanned trial')
    parent=ROOT/plan['parent'];source=ROOT/plan['source']
    if sha(parent)!=plan['parent_sha256'] or sha(source)!=plan['source_sha256']:raise ValueError('Changed lineage')
    output=ROOT/f'runs/p19-{args.arm}-seed{args.seed}';output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    model,state,_=load(parent,'cuda');before=frozen_digest(model,state)
    payload=torch.load(source,map_location='cpu',weights_only=True)
    if args.arm=='recurrent_trace':bridge=RecurrentLanguageBridge(model,plan['recurrent_passes'],payload,plan['parent_sha256']).cuda()
    else:
        bridge=LanguageBridge(model).cuda();load_interface(bridge,payload,plan['parent_sha256'])
    optimizer=torch.optim.AdamW(bridge.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01)
    data=dataset();development=balanced(data['development'],plan['development_examples'],plan['audit_seed']-1)
    documents=read_manifest(ROOT/'runs/chinese-curriculum-v2/manifest.json')
    focus=windows([d for d in documents if d['path'].startswith('focus-')],'train',128)
    basic=windows([d for d in documents if not d['path'].startswith('focus-')],'train',128)
    generator=torch.Generator().manual_seed(2810000+args.seed)
    targets=replay_targets=task_presentations=0;rows=[];train_seconds=0.
    stats={'trainable_parameters':sum(p.numel() for p in bridge.trainable_parameters()),
        'new_parameters':sum(p.numel() for p in bridge.reasoner.parameters()) if args.arm=='recurrent_trace' else 0,
        'plan_sha256':sha(plan_path),'arm':args.arm,'seed':args.seed,'frozen_before':before}
    (output/'config.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for step in range(1,plan['steps_each']+1):
        torch.cuda.synchronize();tick=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if step%plan['replay_every']==0:
            n=plan['batch']//2
            tokens=torch.cat([focus[torch.randint(len(focus),(n,),generator=generator)],
                basic[torch.randint(len(basic),(n,),generator=generator)]]).cuda()
            loss=causal_loss(bridge(tokens,state),tokens)
            replay_targets+=int((tokens[:,1:]!=PAD).sum())
        else:
            indices=torch.randint(len(data['train']),(plan['batch'],),generator=generator).tolist()
            templates=torch.randint(2,(plan['batch'],),generator=generator).tolist()
            tokens,starts=batch([data['train'][i] for i in indices],args.arm,templates)
            tokens=tokens.cuda();starts=starts.cuda()
            loss=response_loss(bridge(tokens,state),tokens,starts)
            targets+=int(((torch.arange(tokens.shape[1],device='cuda')[None]>=starts[:,None])&(tokens!=PAD)).sum())
            task_presentations+=len(indices)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(bridge.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();train_seconds+=time.perf_counter()-tick
        if step%plan['check_every']==0:
            result=audit(bridge,state,development,args.arm)
            result.pop('details')
            unchanged=before==frozen_digest(model,state)
            if not unchanged:raise AssertionError('Old neural state changed')
            row=dict(step=step,loss_last_batch=float(loss.detach()),development=result,old_tensors_exact=unchanged)
            rows.append(row);print(json.dumps(dict(arm=args.arm,seed=args.seed,**row)),flush=True)
            atomic_save({'format':'spora-short-reasoning-v1','organism_sha256':plan['parent_sha256'],
                'arm':args.arm,'passes':plan['recurrent_passes'] if args.arm=='recurrent_trace' else 0,
                'state':interface_state(bridge),'steps':step,'prior_steps':1100,'source_sha256':plan['source_sha256'],
                'plan_sha256':sha(plan_path),'seed':args.seed,'optimizer':optimizer.state_dict(),
                'sampler_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'supervised_task_targets':targets,'replay_targets':replay_targets},output/'checkpoint.pt')
    result=dict(stats,steps=plan['steps_each'],task_presentations=task_presentations,
        task_target_presentations=targets,replay_target_presentations=replay_targets,
        train_seconds=train_seconds,seconds_including_evaluation=time.perf_counter()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        memory_excludes_driver_context=True,checks=rows,old_tensors_exact=before==frozen_digest(model,state),
        checkpoint_sha256=sha(output/'checkpoint.pt'),parent_unchanged=sha(parent)==plan['parent_sha256'],
        source_unchanged=sha(source)==plan['source_sha256'])
    result['passed']=all(result[k] for k in ('old_tensors_exact','parent_unchanged','source_unchanged'))
    (output/'training.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
