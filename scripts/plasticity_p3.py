"""Measure correction and subsequent learning from fixed consolidation candidates."""
import argparse
import json
from pathlib import Path
import torch
from spora.checkpoint import load_checkpoint
from spora.tasks import mappings
from spora.p2 import Event,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError('Preserve evaluation')
    audit=json.loads((Path(args.audit_dir)/'audit.json').read_text())
    model=load_checkpoint(audit['checkpoint'])[0]
    generator=torch.Generator().manual_seed(910000)
    torch.set_num_threads(2);results=[]
    for world in range(4):
        keys,values=mappings(generator,1,24,model.config,'cpu')
        stages=audit['worlds'][world]['stages']
        accepted=[s for s in stages if s['gate']]
        if not accepted:
            results.append({'world':world,'status':'no accepted consolidation'});continue
        stage=accepted[-1]['stage'];n=stage*8
        path=Path(args.audit_dir)/f'world{world}-stage{stage}-candidate.pt'
        student,fast,velocity,_=load_checkpoint(path)
        current_keys=keys[:,:n];current=values[:,:n].clone()
        old_prediction=student.predict_many(current_keys,fast).argmax(-1)
        revisions=[]
        for revision in range(12):
            pos=revision%4;current[:,pos]=(current[:,pos]+5)%model.config.value_count
            event=Event(current_keys[:,pos],current[:,pos],torch.ones(1),'correction')
            fast,velocity,_=write(student,fast,velocity,event,'one')
            with torch.no_grad():pred=student.predict_many(current_keys,fast).argmax(-1)
            revisions.append({'target':bool(pred[0,pos]==current[0,pos]),
                              'all_known':float((pred==current).float().mean())})
        results.append({'world':world,'stage':stage,'facts':n,
                        'before':float((old_prediction==values[:,:n]).float().mean()),
                        'correction_success':sum(r['target'] for r in revisions)/len(revisions),
                        'final_all_known':revisions[-1]['all_known'],'revisions':revisions})
    output.write_text(json.dumps({'audit':args.audit_dir,'results':results},indent=2),encoding='utf-8')
    print(json.dumps({'saved':str(output),'results':results}),flush=True)


if __name__=='__main__':main()
