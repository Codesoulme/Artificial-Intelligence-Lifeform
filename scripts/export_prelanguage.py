"""Export a fixed passing lineage, preserving each transition as separate files."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.temporal_cli import create,learn
from spora.growing import migrate,grow,observe,query,save,load,scalars
from spora.audit_p4 import isolated_query

def main():
    torch.set_num_threads(2)
    for seed in (7,11,23):
        p5=json.loads((ROOT/'runs'/f'p5b-orthogonal90-seed{seed}-audit/audit.json').read_text())
        if not p5['gate']['passed']:raise RuntimeError('P5b prerequisite failed')
        negative=json.loads((ROOT/'runs'/f'p6-negatives-seed{seed}.json').read_text())
        if not negative['passed']:raise RuntimeError('Temporal absence audit failed')
        for entities in (4,64):
            result=json.loads((ROOT/'runs'/f'p6-seed{seed}-entities{entities}/audit.json').read_text())
            if not result['passed']:raise RuntimeError('P6 prerequisite failed')
    source=ROOT/'checkpoints/individual-p5b-candidate.pt';destination=ROOT/'checkpoints/individual-prelanguage.pt'
    if source.exists() or destination.exists():raise FileExistsError('Candidate already exists')
    foundation=ROOT/'checkpoints/p5b-orthogonal90-seed7.pt'
    create(source,foundation)
    for value,trust in ((5,1),(9,1),(3,0)):learn(source,12,value,trust)
    request={'keys':[12,12,12,12,12,13],'modes':[0,1,1,1,1,1],'times':[0,0,1,2,3,0]}
    old=isolated_query(source,**request)
    model,state=migrate(source,destination)
    # Additional pages are explicit future capacity; no new events inserted.
    state=grow(model,state,pages=3)
    with torch.no_grad():new=query(model,state,*[torch.tensor(request[k]) for k in ('keys','modes','times')]).softmax(-1).tolist()
    if new!=old['probabilities']:raise AssertionError('Candidate growth changed queries')
    meta={'stage':'P7 before language training','parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'foundation_sha256':hashlib.sha256(foundation.read_bytes()).hexdigest(),
          'trained_language_tokens':0,'allocation':'four finite neural pages','candidate_gate':'P5b plus P6 fixed audits passed'}
    save(destination,model,state,meta)
    # Continued experience on an in-memory branch verifies future writability; not saved to candidate.
    branch=observe(model,state,12,7,1)
    with torch.no_grad():continued=query(model,branch,torch.tensor([12,12,12,12]),torch.tensor([0,1,1,1]),torch.tensor([0,0,1,2])).argmax(-1).tolist()
    report={'candidate':str(destination),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'source_individual':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'fixed_seed':7,'observations':state.cursor,'pages':len(state.pages),'state_scalars':scalars(state),
            'expected':[9,5,9,3,16,16],'actual':old['predictions'],'adoption_and_growth_exact':True,
            'unsaved_continuation_expected':[7,5,9,3],'unsaved_continuation_actual':continued,
            'basis_buffer_scalars':sum(x.numel() for _,x in model.named_buffers()),
            'trainable_and_frozen_model_parameters':model.parameter_count(),'Chinese_training_started':False}
    if report['actual']!=report['expected'] or continued!=report['unsaved_continuation_expected']:raise AssertionError('Fixed candidate demonstration fails')
    (ROOT/'runs/prelanguage-candidate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
