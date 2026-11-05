"""Small P-only consolidation experiment with whole-model teacher and reset-F tests."""
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


def accuracy(model,keys,values,fast):
    with torch.no_grad():
        return float((model.predict_many(keys,fast).argmax(-1)==values).float().mean())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--seed',type=int,default=710000)
    parser.add_argument('--worlds',type=int,default=3)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--policy',choices=['one','two','adaptive-max4'],default='two')
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'consolidation.json').exists():raise FileExistsError('Preserve previous result')
    torch.set_num_threads(4);generator=torch.Generator().manual_seed(args.seed)
    foundation,_,_,_=load_checkpoint(args.checkpoint,args.device)
    started=time.perf_counter();results=[]
    for world in range(args.worlds):
        # Two sequential groups: transfer first group, learn second, transfer with replay.
        model=copy.deepcopy(foundation)
        keys,values=mappings(generator,1,16,model.config,args.device)
        fast,velocity=model.initial_state(1)
        stages=[]
        for stage in (1,2):
            begin=(stage-1)*8;end=stage*8
            for i in range(begin,end):
                event=Event(keys[:,i],values[:,i],torch.ones(1,device=args.device),'new')
                fast,velocity,_=write(model,fast,velocity,event,args.policy)
            teacher=copy.deepcopy(model);teacher.eval()
            for p in teacher.parameters():p.requires_grad_(False)
            teacher_fast=[[p.detach().clone() for p in site] for site in fast]
            allkeys=keys[:,:end];allvalues=values[:,:end]
            teacher_accuracy=accuracy(teacher,allkeys,allvalues,teacher_fast)
            reset,_=model.initial_state(1)
            before=accuracy(model,allkeys,allvalues,reset)
            fixed={name:p.detach().cpu().clone() for name,p in model.named_parameters() if '.experts.' not in name}
            student=copy.deepcopy(model)
            for name,p in student.named_parameters():p.requires_grad_('.experts.' in name)
            trainable=[p for p in student.parameters() if p.requires_grad]
            optimizer=torch.optim.Adam(trainable,lr=.01)
            sf,su=student.initial_state(1)
            # Teacher's behavior on unrelated vocabulary keys is a preservation anchor.
            used=set(keys[0,:end].tolist())
            anchors=torch.tensor([[k for k in range(model.config.key_count) if k not in used]],device=args.device)
            with torch.no_grad():
                target_prob=teacher.predict_many(allkeys,teacher_fast).softmax(-1)
                anchor_prob=teacher.predict_many(anchors,teacher_fast).softmax(-1)
                anchor_choice=anchor_prob.argmax(-1)
            initial_anchor_agreement=None
            with torch.no_grad():
                initial_anchor_agreement=float((student.predict_many(anchors,sf).argmax(-1)==anchor_choice).float().mean())
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                training_fast=sf if step%2==0 else teacher_fast
                logits=student.predict_many(allkeys,training_fast)
                anchor_logits=student.predict_many(anchors,training_fast)
                verified=F.cross_entropy(logits.flatten(0,1),allvalues.flatten())
                distill=F.kl_div(logits.log_softmax(-1),target_prob,reduction='batchmean')/end
                preserve=F.kl_div(anchor_logits.log_softmax(-1),anchor_prob,reduction='batchmean')/anchors.shape[1]
                loss=verified+.25*distill+.5*preserve
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite consolidation')
                loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step()
            after=accuracy(student,allkeys,allvalues,sf)
            with_original_fast=accuracy(student,allkeys,allvalues,teacher_fast)
            with torch.no_grad():
                preserved=float((student.predict_many(anchors,sf).argmax(-1)==anchor_choice).float().mean())
            untouched=all(torch.equal(v,dict(student.named_parameters())[name].detach().cpu()) for name,v in fixed.items())
            if not untouched:raise AssertionError('Consolidation changed non-P weights')
            # Strict anchor comparison is against the full teacher, not reset baseline.
            passed=(after>=teacher_accuracy-.05 and with_original_fast>=teacher_accuracy-.05
                    and preserved>=.95 and teacher_accuracy>=.95)
            row={'stage':stage,'facts':end,'teacher_with_F':teacher_accuracy,'before_reset_F':before,
                 'after_reset_F':after,'unrelated_teacher_agreement':preserved,
                 'after_preserved_F':with_original_fast,
                 'before_unrelated_agreement':initial_anchor_agreement,'steps':args.steps,
                 'P_only':untouched,'strict_gate':passed,'final_loss':float(loss.detach())}
            # Always keep candidate for review; never erase the actual parent on failure.
            path=output/f'world{world}-stage{stage}-candidate.pt'
            save_checkpoint(output/f'world{world}-stage{stage}-parent.pt',teacher,teacher_fast,velocity,
                            cursor=end,metadata={'stage':'pre-consolidation snapshot','world':world})
            save_checkpoint(path,student,sf,su,cursor=end,metadata={
                'stage':'P2 P-only consolidation candidate','gate':passed,'world':world,
                'parent_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()})
            if world==0:
                restart=isolated_query(path,allkeys.cpu().flatten().tolist(),True)
                with torch.no_grad():expected=student.predict_many(allkeys,sf).argmax(-1).cpu().flatten().tolist()
                row['restart_match']=restart['predictions']==expected
                if not row['restart_match']:raise AssertionError('A-type restart mismatch')
                retained=output/f'world{world}-stage{stage}-retained-fast.pt'
                save_checkpoint(retained,student,teacher_fast,velocity,cursor=end,
                                metadata={'stage':'P2 preserved-F consolidation candidate','gate':passed})
                resumed=isolated_query(retained,allkeys.cpu().flatten().tolist())
                with torch.no_grad():expected_b=student.predict_many(allkeys,teacher_fast).argmax(-1).cpu().flatten().tolist()
                row['retained_fast_restart_match']=resumed['predictions']==expected_b
                if not row['retained_fast_restart_match']:raise AssertionError('B-type restart mismatch')
            stages.append(row);print(json.dumps({'world':world,**row}),flush=True)
            if not passed:
                # Stage 2 is still an explicitly exploratory candidate chain, not publication.
                row['next_stage_status']='candidate continuation only; default unchanged'
            model=student
            for p in model.parameters():p.requires_grad_(True)
            fast,velocity=model.initial_state(1)
        results.append({'world':world,'stages':stages})
    report={'checkpoint':str(Path(args.checkpoint).resolve()),'write_policy':args.policy,
            'worlds':results,'seconds':time.perf_counter()-started,
            'protocol':'Fixed steps, training-only replay; whole-model frozen teacher; P-only candidate',
            'limits':['Finite fact transfer is not semantic generalization','Anchor agreement is behavior preservation, not truth accuracy',
                      'Candidate chains retained even after gate failure; no default individual replaced']}
    (output/'consolidation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'saved':str(output/'consolidation.json'),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
