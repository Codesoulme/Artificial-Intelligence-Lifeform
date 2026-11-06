"""Read-only P22c development diagnostics and reversible path interventions."""
import json
import torch
from train_query import ROOT,dump
from query_task import DEV,wording,evaluation_rows,check,batch
from anchor_labels import target_positions
from ladder_task import evidence_positions
from chinese_curriculum_common import sha
from spora.growing import load
from spora.anchor_query import restore_anchor
from spora.language_training import interface_state,frozen_digest

@torch.no_grad()
def inspect(model,state,rows,template):
    details=[]
    for start in range(0,len(rows),16):
        part=rows[start:start+16];tokens,starts=batch(part,[template]*len(part))
        tokens=tokens.cuda();starts=starts.cuda();model(tokens,state);b=torch.arange(len(part),device='cuda')
        anchor=model.anchor_logits[b,starts-1].argmax(-1).tolist()
        evidence=model.evidence_logits[b,starts-1].argmax(-1).tolist()
        copy=model.copy_gate_logits[b,starts-1,0].sigmoid().tolist()
        expected=target_positions(part,[template]*len(part))
        for r,a,e,c,t in zip(part,anchor,evidence,copy,expected):
            target=evidence_positions([r])[0]-2 if r['indices'] else None
            details.append(dict(known=bool(r['indices']),anchor_correct=a==t,
                evidence_first_byte_correct=e==target if target is not None else None,copy_probability=c))
    known=[r for r in details if r['known']];absent=[r for r in details if not r['known']]
    return dict(anchor_accuracy=sum(r['anchor_correct'] for r in details)/len(details),
                known_evidence_accuracy=sum(r['evidence_first_byte_correct'] for r in known)/len(known),
                known_copy_probability=sum(r['copy_probability'] for r in known)/len(known),
                unknown_copy_probability=sum(r['copy_probability'] for r in absent)/len(absent),details=details)

def main():
    out=ROOT/'runs/p22c-mechanism.json'
    if out.exists():raise FileExistsError(out)
    path=ROOT/'runs/p22c-context-seed7/checkpoint.pt';checkpoint_sha=sha(path)
    prior=json.loads((ROOT/'configs/p20j-content-matching.json').read_text('utf-8'))
    torch.set_num_threads(2);organism,state,_=load(ROOT/prior['parent'],'cuda');before=frozen_digest(organism,state)
    model=restore_anchor(organism,torch.load(path,map_location='cpu',weights_only=True),prior['parent_sha256'],'cuda')
    original=interface_state(model);known,absent=evaluation_rows('development',48);templates=[wording(t) for t in DEV]+[(0,0)]
    result=dict(checkpoint_sha256=checkpoint_sha,scope='48 known/48 absent development cases per wording; same-model interventions, not matched retraining',
        selection={str(t):inspect(model,state,known+absent,t) for t in templates},interventions={})
    for name in ('normal','without_anchor','without_context_residual'):
        with torch.no_grad():
            if name=='without_anchor':model.anchor_out.weight.zero_()
            if name=='without_context_residual':
                model.context_query[-1].weight.zero_();model.context_query[-1].bias.zero_()
        result['interventions'][name]=check(model,state,known,absent,templates)
        merged=model.state_dict();merged.update(original);model.load_state_dict(merged)
    result['interface_restored_exact']=all(torch.equal(t,interface_state(model)[k]) for k,t in original.items())
    result['old_tensors_exact']=before==frozen_digest(organism,state)
    result['checkpoint_unchanged']=sha(path)==checkpoint_sha
    if not all(result[k] for k in ('interface_restored_exact','old_tensors_exact','checkpoint_unchanged')):raise AssertionError('Intervention preservation failed')
    dump(out,result);print(json.dumps({k:{a:b for a,b in v.items() if a!='details'} for k,v in result['selection'].items()}),flush=True)

if __name__=='__main__':main()
