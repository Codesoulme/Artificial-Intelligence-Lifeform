"""Export the preserved original with optional, separately audited capabilities."""
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spora.growing import load,save
from spora.durable import TensorStore
from spora.state_codec import flatten
from audit_prechinese_runtime import protected_all,extend
from audit_durable import fingerprint


def main():
    torch.set_num_threads(2)
    protected_all()
    for name in ('p14-p15-audit-v2','p16-runtime-audit-v2'):
        if not json.loads((ROOT/f'runs/{name}/audit.json').read_text())['passed']:
            raise ValueError('Required gate failed: '+name)
    parent=ROOT/'checkpoints/individual-internal-v2-candidate.pt'
    model,state,meta=extend()
    path=ROOT/'checkpoints/individual-prechinese-v3.pt'
    database=path.with_suffix('.sqlite')
    if path.exists() or database.exists():raise FileExistsError('Never overwrite an individual')
    om,os,_=load(parent)
    old=flatten(om,os)[1];new=flatten(model,state)[1]
    if not all(torch.equal(t,new[k]) for k,t in old.items()):raise AssertionError('Original tensor altered')
    with TensorStore.create(database,model,state,meta) as store:
        rm,rs,_,_=store.read()
        if fingerprint(model,state)!=fingerprint(rm,rs):raise AssertionError('Database mismatch')
        store.checkpoint()
    save(path,model,state,meta)
    rm,rs,_=load(path)
    if fingerprint(model,state)!=fingerprint(rm,rs):raise AssertionError('Portable mismatch')
    report={'passed':True,'original_observations':state.cursor,'old_tensors_exact':True,
        'semantic_active':False,'semantic_parameters':sum(p.numel() for p in model.semantic_memory.parameters()),
        'skill_parameters':sum(p.numel() for p in model.skill_policy.parameters()),
        'tensor_payload_increase':sum(t.numel()*t.element_size() for t in new.values())-sum(t.numel()*t.element_size() for t in old.values()),
        'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (path,database)},
        'note':'Concept mechanism is inactive on the original lineage; no transfer of another synthetic individual facts. Numeric feedback policy is learned and stored.'}
    protected_all()
    (ROOT/'runs/p16-candidate.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
