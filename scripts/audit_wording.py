"""Freeze all P21 candidates; only successful development opens final wording."""
import json
from types import SimpleNamespace
import torch
from train_wording import ROOT, SOURCES, dump, metrics, protected_manifest
from wording_task import TEST, DEVELOPMENT, check, diagnostics, audit
from ladder_task import tasks, sample, check as legacy_check
from chinese_curriculum_common import sha, plan_data, memory_answers
from spora.growing import load
from spora.language_ladder import restore_ladder
from spora.language_training import interface_state, frozen_digest
from spora.language_eval import evaluate
from spora.text_data import windows

def main():
    plan=json.loads((ROOT/'configs/p21-wording.json').read_text('utf-8'))
    prior=json.loads((ROOT/'configs/p20j-content-matching.json').read_text('utf-8'))
    candidates={};configs={}
    for seed in plan['seeds']:
        path=ROOT/f'runs/p21a-seed{seed}'
        training=json.loads((path/'training.json').read_text('utf-8'))
        config=json.loads((path/'config.json').read_text('utf-8'))
        if config['plan']!=plan:raise ValueError('Plan changed')
        if config['source_code_sha256']!={name:sha(name) for name in SOURCES}:raise ValueError('Implementation changed')
        if sha(path/'checkpoint.pt')!=training['checkpoint_sha256']:raise ValueError('Candidate changed')
        if protected_manifest()!=config['protected']:raise ValueError('Protected files changed')
        candidates[str(seed)]=training;configs[str(seed)]=config
    out=ROOT/'runs/p21a-audit';out.mkdir(exist_ok=False)
    dump(out/'frozen-candidates.json',{s:t['checkpoint_sha256'] for s,t in candidates.items()})
    eligible=all(t['status']=='development_passed' for t in candidates.values())
    data=tasks(expanded=True);_,docs,_,_,_=plan_data();torch.set_num_threads(2)
    results=[]
    for seed in plan['seeds']:
        path=ROOT/f'runs/p21a-seed{seed}'
        payload=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        if payload['continuation_source_sha256']!=plan['source_sha256'][str(seed)] or payload['continuation_plan_sha256']!=sha('configs/p21-wording.json'):
            raise ValueError('Continuation lineage differs')
        if payload['continuation_steps']!=candidates[str(seed)]['steps']:raise ValueError('Step counter mismatch')
        organism,state,_=load(ROOT/prior['parent'],'cuda');before=frozen_digest(organism,state)
        model=restore_ladder(organism,payload,prior['parent_sha256'],'cuda')
        restored=interface_state(model)
        exact=all(torch.equal(t,restored[k]) for k,t in payload['state'].items())
        development=check(model,state,data['update']['development'],DEVELOPMENT)
        legacy={s:legacy_check(model,state,d['development'],s) for s,d in data.items()}
        legacy_second=check(model,state,data['update']['development'],((0,1),))
        nll=evaluate(model,state,windows(docs,'development',128))['negative_log_likelihood']
        memory=memory_answers(SimpleNamespace(model=model,state=state))
        row=dict(seed=seed,training=candidates[str(seed)],restore_exact=exact,development=development,legacy=legacy,legacy_second=legacy_second,nll=nll,
                 memory=memory,old_tensors_exact=before==frozen_digest(organism,state),diagnostics=diagnostics(model,state))
        row['development_passed']=min(metrics(development)+metrics(legacy)+metrics(legacy_second))>=plan['gate'] and nll<=prior['source_development_nll']*prior['retention_ratio_max']
        row['engineering_passed']=exact and row['old_tensors_exact'] and memory['passed'] and candidates[str(seed)]['old_tensors_exact'] and all(candidates[str(seed)]['replay_coverage'].values())
        # CPU reload checks a deterministic subset against raw GPU bytes/EOS.
        if seed==7:
            subset=sample(data['update']['development'],16,314000)
            gpu=audit(model,state,subset,DEVELOPMENT[0])
            cpu_organism,cpu_state,_=load(ROOT/prior['parent'],'cpu')
            cpu=restore_ladder(cpu_organism,payload,prior['parent_sha256'],'cpu')
            cpu_outputs=audit(cpu,cpu_state,subset,DEVELOPMENT[0])
            row['cpu_gpu_raw_equal']=[d['output'] for d in gpu['details']]==[d['output'] for d in cpu_outputs['details']]
            row['engineering_passed'] &= row['cpu_gpu_raw_equal']
            del cpu,cpu_state,cpu_organism
        results.append(row);dump(out/f'seed{seed}-development.json',row)
        print(json.dumps(dict(seed=seed,development=development,nll=nll,engineering=row['engineering_passed'],diagnostics={k:v['accuracy'] for k,v in row['diagnostics'].items()})),flush=True)
        del model,organism,state,payload,restored
    eligible &= all(r['development_passed'] and r['engineering_passed'] for r in results)
    # Freeze and recheck EVERY candidate before opening ANY test output.
    if eligible:
        for seed in plan['seeds']:
            if sha(ROOT/f'runs/p21a-seed{seed}/checkpoint.pt')!=candidates[str(seed)]['checkpoint_sha256']:raise ValueError('Candidate changed before test')
        for row in results:
            seed=row['seed'];payload=torch.load(ROOT/f'runs/p21a-seed{seed}/checkpoint.pt',map_location='cpu',weights_only=True)
            organism,state,_=load(ROOT/prior['parent'],'cuda');model=restore_ladder(organism,payload,prior['parent_sha256'],'cuda')
            row['test']=check(model,state,data['update']['test'],TEST,details=True)
            row['legacy_known_test']={s:legacy_check(model,state,d['test'],s) for s,d in data.items()}
            row['legacy_second_known_test']=check(model,state,data['update']['test'],((0,1),))
            row['test_passed']=min(metrics(row['test'])+metrics(row['legacy_known_test'])+metrics(row['legacy_second_known_test']))>=plan['gate']
            dump(out/f'seed{seed}-test.json',row['test'])
            print(json.dumps(dict(seed=seed,test={k:{n:v for n,v in x.items() if n!='records'} for k,x in row['test'].items()},passed=row['test_passed'])),flush=True)
            del model,organism,state,payload
    if protected_manifest()!=configs['7']['protected']:raise ValueError('Protected file changed during audit')
    result=dict(plan=plan,final_test_opened=eligible,passed=eligible and all(r.get('test_passed',False) for r in results),
                protected_files=len(configs['7']['protected']),trials=results)
    dump(out/'audit.json',result)
    print(json.dumps(dict(final_test_opened=eligible,passed=result['passed'])),flush=True)

if __name__=='__main__':main()
