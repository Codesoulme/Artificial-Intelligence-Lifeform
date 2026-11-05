"""Read saved P2 failures and separate argmax changes from true fact loss."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from spora.checkpoint import load_checkpoint
from spora.tasks import mappings
from spora.p2 import Event,write


def main():
    output=Path('runs/p3-diagnosis.json')
    if output.exists():raise FileExistsError('Preserve diagnosis')
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(710000)
    rows=[]
    for world in range(3):
        config=load_checkpoint('checkpoints/p2-v1-delta-seed7.pt')[0].config
        keys,values=mappings(generator,1,16,config,'cpu')
        for stage in (1,2):
            base=Path('runs/p2-consolidation-delta-seed7')
            teacher,tf,tv,_=load_checkpoint(base/f'world{world}-stage{stage}-parent.pt')
            student,sf,sv,_=load_checkpoint(base/f'world{world}-stage{stage}-candidate.pt')
            n=stage*8;known=keys[:,:n];truth=values[:,:n]
            used=set(known[0].tolist())
            anchors=torch.tensor([[k for k in range(config.key_count) if k not in used]])
            with torch.no_grad():
                target=teacher.predict_many(anchors,tf).softmax(-1)
                actual=student.predict_many(anchors,sf).softmax(-1)
                retained=student.predict_many(anchors,tf).softmax(-1)
                disagree=target.argmax(-1)!=actual.argmax(-1)
                margin=target.topk(2).values.diff(dim=-1).abs().squeeze(-1)
                known_score=(student.predict_many(known,sf).argmax(-1)==truth).float().mean()
            changed=(truth[:,0]+5)%config.value_count
            event=Event(known[:,0],changed,torch.ones(1),'correction')
            ff,vv,_=write(student,sf,sv,event,'one')
            with torch.no_grad():
                pred=student.predict_many(known,ff).argmax(-1)
            row={'world':world,'stage':stage,'known_reset_accuracy':float(known_score),
                 'anchor_disagreements':int(disagree.sum()),'anchors':anchors.numel(),
                 'anchor_kl':float(F.kl_div(actual.log(),target,reduction='sum')/anchors.numel()),
                 'changed_teacher_margins':margin[disagree].tolist(),
                 'changed_teacher_confidence':target.max(-1).values[disagree].tolist(),
                 'retained_F_agreement':float((retained.argmax(-1)==target.argmax(-1)).float().mean()),
                 'post_consolidation_one_write_correction':int(pred[0,0]==changed[0]),
                 'post_correction_other_known':float((pred[:,1:]==truth[:,1:]).float().mean())}
            rows.append(row)
    result={'source':'Read-only saved P2 candidates; future audit remains unused','rows':rows}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
