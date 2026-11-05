"""Bounded consolidation experiments with preservation and future-plasticity gates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .checkpoint import load_checkpoint,save_checkpoint
from .tasks import mappings
from .p2 import Event,write
from .evaluate import isolated_query


def clone_fast(fast):
    return [[p.detach().clone().requires_grad_(True) for p in site] for site in fast]


def rebase_delta(model,fast,velocity):
    """Exact finite-state transfer into existing initial matrices, not compression.

    For the direct delta writer initial matrices are not used during updates;
    copying the live matrices into those parameters preserves all subsequent
    queries and direct writes. Resetting F then restores the transferred state.
    Parameter count stays fixed; information capacity does not increase.
    """
    if model.config.memory_kind!='delta' or model.config.residual_write:
        raise ValueError('Exact reference requires the direct delta writer')
    if len(fast)!=len(model.memories) or len(velocity)!=len(fast):raise ValueError('Bad state')
    for site,params,moments in zip(model.memories,fast,velocity):
        if len(params)!=1 or len(moments)!=1 or params[0].shape!=(1,*site.initial[0].shape):
            raise ValueError('Transfer requires exactly one individual')
        if not bool(torch.isfinite(params[0]).all()) or not bool(torch.isfinite(moments[0]).all()):
            raise ValueError('Nonfinite state')
        if bool(torch.count_nonzero(moments[0])):raise ValueError('Nonzero delta velocity')
    student=copy.deepcopy(model)
    with torch.no_grad():
        for site,params in zip(student.memories,fast):site.initial[0].copy_(params[0][0])
    sf,sv=student.initial_state(1)
    return student,sf,sv


def accuracy(model,keys,values,fast):
    if not keys.numel():return None
    with torch.no_grad():return float((model.predict_many(keys,fast).argmax(-1)==values).float().mean())


def kl(logits,prob):
    return F.kl_div(logits.log_softmax(-1),prob,reduction='sum')/prob.shape[-2]


def fit_experts(teacher,fast,keys,values,steps,anchor_kl,anchor_margin):
    student=copy.deepcopy(teacher)
    for name,p in student.named_parameters():p.requires_grad_('.experts.' in name)
    trainable=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(trainable,lr=.01)
    sf,sv=student.initial_state(1)
    sf=[[p.detach() for p in site] for site in sf]
    tf=[[p.detach() for p in site] for site in fast]
    used=set(keys[0].tolist())
    anchors=torch.tensor([[k for k in range(student.config.key_count) if k not in used]],device=keys.device)
    with torch.no_grad():
        known_prob=teacher.predict_many(keys,fast).softmax(-1)
        anchor_prob=teacher.predict_many(anchors,fast).softmax(-1)
        choices=anchor_prob.argmax(-1)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        state=sf if step%2==0 else tf
        logits=student.predict_many(keys,state)
        anchor_logits=student.predict_many(anchors,state)
        chosen=anchor_logits.gather(-1,choices[...,None]).squeeze(-1)
        other=anchor_logits.masked_fill(F.one_hot(choices,student.config.value_count).bool(),float('-inf')).max(-1).values
        margin=F.relu(.15+other-chosen).mean()
        loss=F.cross_entropy(logits.flatten(0,1),values.flatten())+.25*kl(logits,known_prob)
        loss=loss+anchor_kl*kl(anchor_logits,anchor_prob)+anchor_margin*margin
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
    untouched=all(torch.equal(p,dict(teacher.named_parameters())[name])
                  for name,p in student.named_parameters() if '.experts.' not in name)
    if not untouched:raise AssertionError('Non-expert weights changed')
    for p in student.parameters():p.requires_grad_(True)
    sf,sv=student.initial_state(1)
    return student,sf,sv,{'steps':steps,'P_only':True,'loss':float(loss.detach())}


def plasticity(model,fast,velocity,keys,values):
    successes=[];retention=[]
    for index in range(min(4,keys.shape[1])):
        changed=(values[:,index]+(index*3+5))%model.config.value_count
        event=Event(keys[:,index],changed,torch.ones(1,device=keys.device),'correction')
        ff,vv,_=write(model,clone_fast(fast),[[u.clone() for u in site] for site in velocity],event,'one')
        target=values.clone();target[:,index]=changed
        with torch.no_grad():pred=model.predict_many(keys,ff).argmax(-1)
        successes.append(float((pred[:,index]==changed).float().mean()))
        mask=torch.ones_like(values,dtype=torch.bool);mask[:,index]=False
        retention.append(float((pred[mask]==values[mask]).float().mean()))
    return {'correction':sum(successes)/len(successes),'post_correction_unrelated':sum(retention)/len(retention)}


def assess(teacher,teacher_fast,student,fast,velocity,keys,values,old_count):
    used=set(keys[0].tolist())
    anchors=torch.tensor([[k for k in range(student.config.key_count) if k not in used]],device=keys.device)
    reset,_=student.initial_state(1)
    with torch.no_grad():
        t=teacher.predict_many(anchors,teacher_fast).softmax(-1)
        a=student.predict_many(anchors,reset)
        b=student.predict_many(anchors,teacher_fast)
    row={'teacher_known':accuracy(teacher,keys,values,teacher_fast),
         'known_reset':accuracy(student,keys,values,reset),
         'known_retained':accuracy(student,keys,values,teacher_fast),
         'old_known_reset':accuracy(student,keys[:,:old_count],values[:,:old_count],reset),
         'new_known_reset':accuracy(student,keys[:,old_count:],values[:,old_count:],reset),
         'anchor_reset':float((a.argmax(-1)==t.argmax(-1)).float().mean()),
         'anchor_retained':float((b.argmax(-1)==t.argmax(-1)).float().mean()),
         'anchor_kl_reset':float(kl(a,t)),
         **plasticity(student,fast,velocity,keys,values)}
    gate_keys=('teacher_known','known_reset','known_retained','anchor_reset','anchor_retained',
               'correction','post_correction_unrelated')
    row['weakest']=min(row[k] for k in gate_keys)
    row['gate']=all(row[k]>=.95 for k in gate_keys)
    return row


def chain(checkpoint,seed,worlds,stages,steps,variant,writer,output,device,stop_on_failure=False,restart=False):
    foundation,_,_,_=load_checkpoint(checkpoint,device)
    generator=torch.Generator().manual_seed(seed)
    records=[];started=time.perf_counter()
    for world in range(worlds):
        model=copy.deepcopy(foundation);fast,velocity=model.initial_state(1)
        keys,values=mappings(generator,1,stages*8,model.config,device)
        rows=[]
        for stage in range(1,stages+1):
            begin=(stage-1)*8;end=stage*8
            for i in range(begin,end):
                event=Event(keys[:,i],values[:,i],torch.ones(1,device=device),'new')
                fast,velocity,_=write(model,fast,velocity,event,'one')
            teacher=copy.deepcopy(model);tf=clone_fast(fast)
            for p in teacher.parameters():p.requires_grad_(False)
            if variant['name']=='exact_rebase':
                student,sf,sv=rebase_delta(model,fast,velocity);fit={'steps':0,'exact_transfer':True}
            else:
                student,sf,sv,fit=fit_experts(teacher,tf,keys[:,:end],values[:,:end],steps,
                                           variant['anchor_kl'],variant['anchor_margin'])
                student.config.residual_write=writer=='residual'
            row={'world':world,'stage':stage,'facts':end,'variant':variant['name'],'writer':writer,
                 **fit,**assess(teacher,tf,student,sf,sv,keys[:,:end],values[:,:end],begin)}
            if output is not None:
                path=output/f'world{world}-stage{stage}-candidate.pt'
                save_checkpoint(output/f'world{world}-stage{stage}-parent.pt',model,fast,velocity,
                                cursor=end,metadata={'stage':'P3 pre-consolidation snapshot','world':world})
                save_checkpoint(path,student,sf,sv,cursor=end,metadata={
                    'stage':'P3 research candidate','gate':row['gate'],'variant':variant['name'],
                    'parent_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()})
                if restart and world==0:
                    expected=student.predict_many(keys[:,:end],sf).argmax(-1).cpu().flatten().tolist()
                    resumed=isolated_query(path,keys[:,:end].cpu().flatten().tolist(),True)['predictions']
                    row['restart_match']=expected==resumed
                    if not row['restart_match']:raise AssertionError('Restart mismatch')
                    retained_path=output/f'world{world}-stage{stage}-retained-fast.pt'
                    save_checkpoint(retained_path,student,tf,velocity,cursor=end,
                                    metadata={'stage':'P3 retained-fast verification','gate':row['gate']})
                    retained_expected=student.predict_many(keys[:,:end],tf).argmax(-1).cpu().flatten().tolist()
                    retained_resumed=isolated_query(retained_path,keys[:,:end].cpu().flatten().tolist())['predictions']
                    row['retained_restart_match']=retained_expected==retained_resumed
                    if not row['retained_restart_match']:raise AssertionError('Retained-state restart mismatch')
            rows.append(row);print(json.dumps(row),flush=True)
            if stop_on_failure and not row['gate']:
                row['chain_action']='rejected: keep parent; stop candidate chain';break
            row['chain_action']='candidate continuation only' if not row['gate'] else 'accepted within isolated experiment'
            model=student;fast,velocity=sf,sv
        records.append({'world':world,'stages':rows})
    return {'checkpoint':str(Path(checkpoint).resolve()),'variant':variant,'writer':writer,
            'worlds':records,'seconds':time.perf_counter()-started}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['development','audit'],required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--selection')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();torch.set_num_threads(2)
    plan=json.loads(Path('configs/p3-plan.json').read_text())
    output=Path(args.output_dir)
    if output.exists():raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    if args.phase=='development':
        candidates=[]
        for variant in plan['development_variants']:
            for writer in ('direct','residual'):
                dest=output/(variant['name']+'-'+writer);dest.mkdir()
                result=chain(args.checkpoint,plan['development_seed'],plan['development_worlds'],
                             2,plan['steps_per_stage'],variant,writer,dest,args.device)
                rows=[r for w in result['worlds'] for r in w['stages']]
                result['score']=[sum(r['gate'] for r in rows),sum(r['weakest'] for r in rows)/len(rows)]
                candidates.append(result)
                (dest/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        winner=max(candidates,key=lambda r:r['score'])
        result={'phase':'development','checkpoint':args.checkpoint,
                'sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                'selected':{'variant':winner['variant'],'writer':winner['writer']},'candidates':candidates}
    else:
        selection_path=Path(args.selection)
        locked_path=Path('runs/p3-selection-lock.json')
        if not locked_path.exists():raise ValueError('Run scripts.audit_p3 to seal development selection first')
        locked=json.loads(locked_path.read_text(encoding='utf-8'))
        selection_digest=hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if selection_digest!=locked['selection_sha256']:raise ValueError('Development selection changed')
        selection=json.loads(selection_path.read_text(encoding='utf-8'))['selected']
        result=chain(args.checkpoint,plan['audit_seed'],plan['audit_worlds_per_foundation'],
                     plan['stages'],plan['steps_per_stage'],**selection,output=output,device=args.device,
                     stop_on_failure=True,restart=True)
        result['phase']='audit';result['selection_source']=args.selection
        result['selection_sha256']=selection_digest
    (output/(args.phase+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'saved':str(output),'phase':args.phase}),flush=True)


if __name__=='__main__':main()
