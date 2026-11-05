"""Export preserved original lineage, plus a separate bounded sleep continuation."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.durable import TensorStore,adopt
from spora.growing import observe,query,save,scalars
from spora.sleep_memory import consolidate_store
from audit_sleep import metrics
from audit_durable import fingerprint


def main():
    torch.set_num_threads(2)
    for path in ('runs/p9-durable-audit/audit.json','runs/p10-sleep-audit/audit.json','runs/p10-sleep-transaction/audit.json'):
        if not json.loads((ROOT/path).read_text())['passed']:raise RuntimeError('Required gate failed')
    source=ROOT/'checkpoints/individual-prelanguage.pt'
    preserved=ROOT/'checkpoints/individual-durable.sqlite'
    sleeping=ROOT/'checkpoints/individual-sleep-candidate.sqlite'
    portable=ROOT/'checkpoints/individual-sleep-candidate.pt'
    if any(p.exists() for p in (preserved,sleeping,portable)):raise FileExistsError('Candidate already exists')
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    with adopt(source,preserved) as original:
        model,state,_,_=original.read();initial_fingerprint=fingerprint(model,state)
        original.backup(sleeping);original.checkpoint()
    # This branch continues the actual three-event individual in a disclosed
    # synthetic environment. Keep key 12 untouched until a final correction.
    keys,values,trusts=[12,12,12],[5,9,3],[1,1,0]
    generator=torch.Generator().manual_seed(2219000)
    with TensorStore(sleeping) as store:
        model,state,_,_=store.read()
        assert fingerprint(model,state)==initial_fingerprint
        for _ in range(61):
            key=int(torch.randint(63,(1,),generator=generator));key=key+(key>=12)
            value=int(torch.randint(16,(1,),generator=generator))
            trust=int(torch.rand(1,generator=generator)>.25)
            state=observe(model,state,key,value,trust)
            keys.append(key);values.append(value);trusts.append(trust)
            store.commit(model,state,store.revision)
        before=scalars(state)
        selection=json.loads((ROOT/'configs/p10-selection.json').read_text())
        sleep=consolidate_store(store,model,state,store.revision,0,selection['rank'],selection['steps'],2219000)
        if not sleep['accepted']:raise RuntimeError('Demonstration candidate rejected; preserve files for inspection')
        model,state,_,_=store.read()
        after=scalars(state)
        for i in range(8):
            key,value,trust=(12,7,1) if i==7 else (i,(i+3)%16,1)
            state=observe(model,state,key,value,trust)
            keys.append(key);values.append(value);trusts.append(trust)
            store.commit(model,state,store.revision)
        evaluation=metrics(model,state,keys,values,trusts)
        if any(x is not None and x<.95 for x in evaluation.values()):raise RuntimeError('Candidate continuation failed')
        with torch.no_grad():
            answers=query(model,state,torch.tensor([12]*4),torch.tensor([0,1,1,1]),torch.tensor([0,0,1,2])).argmax(-1).tolist()
        if answers!=[7,5,9,3]:raise AssertionError('Original personal history changed')
        metadata=dict(store.metadata,source_individual_sha256=source_hash,stage='P10 sleep continuation',
                      generated_environment_seed=2219000,Chinese_training_started=False)
        store.commit(model,state,store.revision,metadata)
        save(portable,model,state,metadata);store.compact()
        final_revision=store.revision
    report={'original_source':str(source),'source_sha256':source_hash,
            'preserved_database':str(preserved),'sleep_candidate':str(sleeping),'portable_candidate':str(portable),
            'observations':72,'original_observations':3,'additional_synthetic_observations':69,
            'sleep':sleep,'before_state':before,'after_state':after,'final_metrics':evaluation,
            'demonstration':[7,5,9,3],'revision':final_revision,
            'files':{p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (preserved,sleeping,portable)},
            'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,
            'Chinese_training_started':False}
    (ROOT/'runs/p9-p10-candidates.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
