"""P22 frozen-candidate audit; final wordings remain closed after any failed gate."""
import argparse,json
from types import SimpleNamespace
import torch
from train_query import ROOT,SOURCES,dump,protection,metrics
from query_task import TEST,DEV,wording,check,evaluation_rows,gate_values
from wording_task import audit,check as short_check,DEVELOPMENT as SHORT_DEV,TEST as SHORT_TEST
from ladder_task import tasks,check as legacy_check
from chinese_curriculum_common import sha,plan_data,memory_answers
from spora.growing import load
from spora.query_bridge import restore_query
from spora.anchor_query import restore_anchor
from spora.language_training import interface_state,frozen_digest
from spora.language_eval import evaluate
from spora.text_data import windows

@torch.no_grad()
def anchor_accuracy(model,state,rows,templates):
    from scripts.anchor_labels import target_positions
    from scripts.query_task import batch
    result={}
    for template in templates:
        correct=0
        for start in range(0,len(rows),16):
            subset=rows[start:start+16];wordings=[template]*len(subset)
            tokens,starts=batch(subset,wordings);device=next(model.parameters()).device
            model(tokens.to(device),state)
            predicted=model.anchor_logits[torch.arange(len(subset),device=device),starts.to(device)-1].argmax(-1).tolist()
            correct+=sum(a==b for a,b in zip(predicted,target_positions(subset,wordings)))
        result[str(template)]=dict(count=len(rows),accuracy=correct/len(rows))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/p22-query.json');p.add_argument('--tag',default='p22a')
    args=p.parse_args();plan=json.loads((ROOT/args.plan).read_text('utf-8'))
    prior=json.loads((ROOT/'configs/p20j-content-matching.json').read_text('utf-8'))
    candidates=[];protected=protection()
    for seed in plan['seeds']:
        for arm in plan['arms']:
            folder=ROOT/f'runs/{args.tag}-{arm}-seed{seed}'
            if not (folder/'training.json').exists():continue
            training=json.loads((folder/'training.json').read_text('utf-8'));config=json.loads((folder/'config.json').read_text('utf-8'))
            if config['plan']!=plan or config['protected']!=protected:raise ValueError('Configuration or protected files changed')
            if any(sha(name)!=value for name,value in config['source_code_sha256'].items()):raise ValueError('Training source changed')
            if sha(folder/'checkpoint.pt')!=training['checkpoint_sha256']:raise ValueError('Candidate changed')
            candidates.append(dict(seed=seed,arm=arm,training=training,configuration=config,folder=str(folder.relative_to(ROOT)),baseline=json.loads((folder/'baseline.json').read_text('utf-8'))))
    if not candidates:raise ValueError('No completed candidates')
    out=ROOT/f'runs/{args.tag}-audit';out.mkdir(exist_ok=False)
    dump(out/'frozen-candidates.json',{r['folder']:r['training']['checkpoint_sha256'] for r in candidates})
    _,docs,_,_,_=plan_data();data=tasks(expanded=True);torch.set_num_threads(2)
    eligible=len([r for r in candidates if r['arm']=='context' and r['training']['status']=='development_passed'])==len(plan['seeds'])
    results=[]
    # Reload all candidates before opening final wordings. Failed pilot gets
    # a development-only audit and remains explicitly incomplete capability.
    for row in candidates:
        path=ROOT/row['folder'];payload=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        if payload['plan_sha256']!=sha(args.plan) or payload['source_sha256']!=plan['source_sha256'][str(row['seed'])]:raise ValueError('Lineage differs')
        if payload['steps']!=row['training']['steps'] or payload['arm']!=row['arm'] or payload['seed']!=row['seed']:raise ValueError('Identity differs')
        organism,state,_=load(ROOT/prior['parent'],'cuda');before=frozen_digest(organism,state)
        restore=restore_anchor if payload['format']=='spora-anchor-query-v1' else restore_query
        model=restore(organism,payload,prior['parent_sha256'],'cuda')
        actual=interface_state(model);row['restore_exact']=all(torch.equal(v,actual[k]) for k,v in payload['state'].items())
        known,absent=evaluation_rows('development',864 if eligible else 96)
        row['development']=check(model,state,known,absent,[wording(t) for t in DEV]+[(0,0)],details=True)
        if hasattr(model,'anchor_logits'):
            row['anchor_location']=anchor_accuracy(model,state,known+absent,[wording(t) for t in DEV]+[(0,0)])
        row['legacy']={s:legacy_check(model,state,d['development'],s) for s,d in data.items()}
        row['short']=short_check(model,state,known,SHORT_DEV+SHORT_TEST+((0,1),))
        row['nll']=evaluate(model,state,windows(docs,'development',128))['negative_log_likelihood']
        row['memory']=memory_answers(SimpleNamespace(model=model,state=state))
        row['old_tensors_exact']=before==frozen_digest(organism,state)
        row['engineering_passed']=row['restore_exact'] and row['old_tensors_exact'] and row['memory']['passed'] and all(row['training']['replay_coverage'].values())
        row['development_passed']=min(gate_values(row['development'])+metrics(row['legacy'])+metrics(row['short']))>=plan['gate'] and row['nll']<=prior['source_development_nll']*prior['retention_ratio_max']
        subset=known[:8]+absent[:8];gpu=audit(model,state,subset,wording(DEV[0]))
        cpu_organism,cpu_state,_=load(ROOT/prior['parent'],'cpu');cpu=restore(cpu_organism,payload,prior['parent_sha256'],'cpu')
        cpu_result=audit(cpu,cpu_state,subset,wording(DEV[0]))
        row['cpu_gpu_raw_equal']=[d['output'] for d in gpu['details']]==[d['output'] for d in cpu_result['details']]
        row['engineering_passed'] &= row['cpu_gpu_raw_equal']
        if row['arm']=='context':eligible &= row['development_passed'] and row['engineering_passed']
        row['checkpoint_bytes']=(path/'checkpoint.pt').stat().st_size
        results.append(row);dump(out/f"{row['arm']}-seed{row['seed']}.json",row)
        print(json.dumps(dict(seed=row['seed'],arm=row['arm'],development={k:{a:b for a,b in v.items() if a!='records'} for k,v in row['development'].items()},engineering=row['engineering_passed'])),flush=True)
        del model,organism,state,payload,actual,cpu,cpu_organism,cpu_state
    if eligible:
        for row in results:
            if sha(ROOT/row['folder']/'checkpoint.pt')!=row['training']['checkpoint_sha256']:raise ValueError('Candidate changed before final')
        for row in results:
            payload=torch.load(ROOT/row['folder']/'checkpoint.pt',map_location='cpu',weights_only=True)
            organism,state,_=load(ROOT/prior['parent'],'cuda')
            restore=restore_anchor if payload['format']=='spora-anchor-query-v1' else restore_query
            model=restore(organism,payload,prior['parent_sha256'],'cuda')
            known,absent=evaluation_rows('test',864)
            row['test']=check(model,state,known,absent,[wording(t) for t in TEST]+[(0,0)],details=True)
            row['test_passed']=min(gate_values(row['test']))>=plan['gate']
            print(json.dumps(dict(seed=row['seed'],arm=row['arm'],test_passed=row['test_passed'])),flush=True)
            del model,organism,state,payload
    schedules={}
    for seed in plan['seeds']:
        arms=[r for r in results if r['seed']==seed]
        if len(arms)>1:
            schedules[str(seed)]=len({r['training']['schedule_sha256'] for r in arms})==1
    if not all(schedules.values()) or protection()!=protected:raise ValueError('Control data/protection mismatch')
    result=dict(plan=plan,final_test_opened=bool(eligible),passed=bool(eligible) and all(r.get('test_passed',False) for r in results if r['arm']=='context'),
                matched_schedules=schedules,protected_files=len(protected),trials=results)
    dump(out/'audit.json',result);print(json.dumps(dict(final_test_opened=result['final_test_opened'],passed=result['passed'])),flush=True)

if __name__=='__main__':main()
