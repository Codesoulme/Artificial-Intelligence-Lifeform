"""Publish an untouched-lineage copy after independent gates have passed."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.growing import load,save
from spora.internal_memory import attach
from spora.durable import TensorStore
from spora.state_codec import flatten
from audit_internal_memory import controller
from audit_internal_runtime import protected
from audit_time_runtime import history_equal
from audit_durable import fingerprint


def main():
    torch.set_num_threads(2);protected()
    for name in ('runs/p12-internal-audit-v2/audit.json','runs/p12-runtime-audit-v2/audit.json'):
        if not json.loads((ROOT/name).read_text())['passed']:raise RuntimeError('Internal audit failed')
    routing=json.loads((ROOT/'runs/p13-read-audit/audit.json').read_text())
    if not routing['correctness_passed']:raise RuntimeError('Router output check failed')
    parent=ROOT/'checkpoints/individual-time-candidate.pt'
    database=ROOT/'checkpoints/individual-internal-v2-candidate.sqlite';portable=database.with_suffix('.pt')
    if any(p.exists() for p in (database,portable)):raise FileExistsError('Candidate already exists')
    model,state,meta=load(parent);old_model,old_state,_=load(parent);control,training=controller(7)
    state=attach(model,state,control)
    meta=dict(meta,stage='P12 internal provenance and learned write control',
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),write_training=training,
        Chinese_training_started=False,default_read='exact',legacy_provenance='unknown',
        source_quality_note='Declared numerical environment inputs; not independent truth authentication')
    old_tensors=flatten(old_model,old_state)[1];schema,tensors=flatten(model,state)
    old_exact=all(torch.equal(v,tensors[k]) for k,v in old_tensors.items())
    same_history=history_equal(old_model,old_state,model,state)
    if not old_exact or not same_history:raise AssertionError('Existing individual changed during adoption')
    with TensorStore.create(database,model,state,meta) as store:
        m,s,_,_=store.read()
        if fingerprint(m,s)!=fingerprint(model,state):raise AssertionError('Database roundtrip failed')
        store.checkpoint()
    save(portable,model,state,meta);m,s,_=load(portable)
    if fingerprint(m,s)!=fingerprint(model,state):raise AssertionError('Portable mismatch')
    protected()
    report={'observations':state.cursor,'provenance_start':state.internal.start_cursor,'source_observations':0,
        'write_parameters':sum(p.numel() for p in control.parameters()),'old_tensors_exact':old_exact,'old_history_exact':same_history,
        'tensor_bytes_before':sum(t.numel()*t.element_size() for t in old_tensors.values()),
        'tensor_bytes_after':sum(t.numel()*t.element_size() for t in tensors.values()),
        'parent_file_bytes':parent.stat().st_size,
        'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (database,portable)},
        'default_read':'exact','learned_router_prepared':False,'passed':True}
    (ROOT/'runs/p12-v2-candidate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
