"""Development-only inference interventions; never retrain or alter a checkpoint."""
import argparse,json
import torch
from ladder_task import ROOT,tasks,check,evidence_audit
from audit_short_reasoning import sha
from spora.growing import load
from spora.language_ladder import restore_ladder
from spora.language_training import frozen_digest,interface_state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'))
    torch.set_num_threads(2)
    parent=ROOT/plan['parent'];path=ROOT/args.checkpoint
    if sha(parent)!=plan['parent_sha256']:raise ValueError('Parent changed')
    before_file=sha(path);payload=torch.load(path,map_location='cpu',weights_only=True)
    if not payload.get('local_evidence'):raise ValueError('Local evidence candidate required')
    organism,state,_=load(parent,args.device);before=frozen_digest(organism,state)
    model=restore_ladder(organism,payload,plan['parent_sha256'],args.device)
    data=tasks(expanded=plan.get('expanded_objects',False))['update']['development']
    results={}
    recency=model.recency_logit.detach().clone()
    for mode in ('normal','no_copy','no_recency'):
        model.copy_reader=mode!='no_copy'
        with torch.no_grad():model.recency_logit.copy_(recency if mode!='no_recency' else torch.full_like(recency,-100.))
        results[mode]={'metrics':check(model,state,data,'update',True)}
        print(json.dumps({'mode':mode,**{k:v for k,v in results[mode]['metrics'].items() if k!='records'}}),flush=True)
    model.copy_reader=True
    with torch.no_grad():model.recency_logit.copy_(recency)
    results['evidence_location']=evidence_audit(model,state,data)
    restored=interface_state(model)
    if any(not torch.equal(v,restored[k]) for k,v in payload['state'].items()):raise AssertionError('Intervention did not restore tensors')
    if before!=frozen_digest(organism,state) or sha(path)!=before_file:raise AssertionError('Protected state changed')
    result={'checkpoint':args.checkpoint,'checkpoint_sha256':before_file,'device':args.device,
        'scope':'complete update development partition; inference-only interventions on one trained model, not matched retrained controls',
        'seed':payload['seed'],'learned_recency_strength':float(torch.nn.functional.softplus(recency)),
        'results':results,'state_and_file_preserved':True}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
