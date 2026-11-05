"""Evaluate frozen selected P20 descendants on the complete held-out pair split."""
import argparse,json
from types import SimpleNamespace
import torch
from ladder_task import ROOT,tasks,check,evidence_audit
from chinese_curriculum_common import plan_data,content_probes,memory_answers
from audit_short_reasoning import protected_files,sha,latency
from spora.growing import load
from spora.language_ladder import restore_ladder
from spora.language_training import interface_state,frozen_digest
from spora.language_eval import evaluate
from spora.text_data import windows
from scripts.reasoning_task import audit,changed_query,reversed_events


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['original','cortex'],default='cortex');p.add_argument('--output',default='runs/p20-final-audit')
    p.add_argument('--plan',default='configs/p20-language-ladder.json');p.add_argument('--tag',default='p20')
    args=p.parse_args();out=ROOT/args.output
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));protected=protected_files(plan)
    # Freeze every candidate before opening any final-test output. A later
    # missing/failed seed must not cause a partial test-driven selection cycle.
    candidates={}
    for seed in plan['seeds']:
        path=ROOT/f'runs/{args.tag}-{args.arm}-seed{seed}'
        training=json.loads((path/'training.json').read_text('utf-8'))
        config=json.loads((path/'config.json').read_text('utf-8'))
        if training['status']!='development_passed':raise ValueError('All seeds must pass development before testing')
        if config['plan']!=plan:raise ValueError('Training plan differs from frozen audit plan')
        if any(sha(name)!=digest for name,digest in config['source_code_sha256'].items()):
            raise ValueError('Training implementation changed since candidate creation')
        if sha(path/'checkpoint.pt')!=training['checkpoint_sha256']:raise ValueError('Candidate file changed')
        candidates[str(seed)]=training['checkpoint_sha256']
    out.mkdir(exist_ok=False)
    (out/'frozen-candidates.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
    _,docs,_,_,_=plan_data();data=tasks(expanded=plan.get('expanded_objects',False));torch.set_num_threads(2)
    probes=[]
    for name in ('values-probes.json','values-probes-v2.json'):probes+=json.loads((ROOT/'data/chinese-pilot'/name).read_text('utf-8'))['probes']
    result={'plan':plan,'frozen_candidates':candidates,'protected_hashes':protected,'trials':[],'test_scope':'complete held-out P19 pair partition; known benchmark, not secret newly collected data'}
    # The second wording was also trained. Check its complete development
    # partition for every seed before opening final test for any seed.
    pretest=[]
    for seed in plan['seeds']:
        path=ROOT/f'runs/{args.tag}-{args.arm}-seed{seed}'
        payload=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        organism,state,_=load(ROOT/plan['parent'],'cuda')
        model=restore_ladder(organism,payload,plan['parent_sha256'],'cuda')
        rows=data['update']['development'];standard=audit(model,state,rows,'answer',1)
        item={'seed':seed,'standard':standard,'pairs':{}}
        for name,transform in (('changed_query',changed_query),('reversed_events',reversed_events)):
            changed=audit(model,state,[transform(r) for r in rows],'answer',1)
            item['pairs'][name]={'results':changed,'both_correct':sum(a['answer_correct'] and b['answer_correct'] for a,b in zip(standard['details'],changed['details']))/len(rows)}
        item['passed']=min([standard['answer_accuracy']]+[v['both_correct'] for v in item['pairs'].values()])>=.95
        pretest.append(item)
        print(json.dumps({'pretest_development_seed':seed,'answer':standard['answer_accuracy'],'passed':item['passed']}),flush=True)
        del model,organism,state,payload
    (out/'pretest-development.json').write_text(json.dumps(pretest,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(item['passed'] for item in pretest):raise ValueError('Second wording development failed; final test remains unopened')
    result['pretest_development']=pretest
    for seed in plan['seeds']:
        path=ROOT/f'runs/{args.tag}-{args.arm}-seed{seed}'
        training=json.loads((path/'training.json').read_text('utf-8'))
        if training['status']!='development_passed':raise ValueError('Do not open test before development passed')
        payload=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        if sha(path/'checkpoint.pt')!=training['checkpoint_sha256'] or payload['plan_sha256']!=sha(args.plan):raise ValueError('Changed final candidate')
        if payload['seed']!=seed or payload['arm']!=args.arm or payload['steps']!=training['steps'] or payload['source_sha256']!=plan['source_sha256']:
            raise ValueError('Trial lineage mismatch')
        if payload.get('depth',2)!=plan.get('depth',2) or payload.get('local_bytes',False)!=plan.get('local_bytes',False) or payload.get('evidence_reader',False)!=plan.get('evidence_reader',False):
            raise ValueError('Trial structure mismatch')
        if payload.get('copy_reader',False)!=plan.get('copy_reader',False):raise ValueError('Copy structure mismatch')
        if payload.get('local_evidence',False)!=plan.get('local_evidence',False):raise ValueError('Local evidence mismatch')
        if payload.get('query_window',32)!=plan.get('query_window',32):raise ValueError('Query window mismatch')
        organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
        model=restore_ladder(organism,payload,plan['parent_sha256'],'cuda')
        restored=interface_state(model);exact=all(torch.equal(v,restored[k]) for k,v in payload['state'].items())
        row={'seed':seed,'arm':args.arm,'training':training,'cold_restore_exact':exact,
            'stages':{stage:check(model,state,parts['test'],stage,True) for stage,parts in data.items()},
            'full_development':{stage:check(model,state,parts['development'],stage) for stage,parts in data.items()},
            'second_training_wording':audit(model,state,data['update']['test'],'answer',1),
            'new_wording':audit(model,state,data['update']['test'],'answer',2),
            'language_development':evaluate(model,state,windows(docs,'development',128)),
            'language_test':evaluate(model,state,windows(docs,'test',128)),
            'focus':evaluate(model,state,windows([d for d in docs if d['path'].startswith('focus-')],'train',128)),
            'values_probes':content_probes(model,state,probes),'memory':memory_answers(SimpleNamespace(model=model,state=state)),
            'checkpoint_bytes':(path/'checkpoint.pt').stat().st_size}
        row['latency']=latency(model,state,data['update']['test'][0])
        row['evidence_location']=evidence_audit(model,state,data['update']['test'])
        row['second_wording_pairs']={}
        original=row['second_training_wording']['details']
        for name,transform in (('changed_query',changed_query),('reversed_events',reversed_events)):
            changed=audit(model,state,[transform(r) for r in data['update']['test']],'answer',1)
            row['second_wording_pairs'][name]={'results':changed,
                'both_correct':sum(a['answer_correct'] and b['answer_correct'] for a,b in zip(original,changed['details']))/len(original)}
        # Independent reverse scan validates both label generation and the
        # evaluator on every persisted standard and counterfactual output.
        independent=True
        for stage in row['stages'].values():
            for record in stage['records'].values():
                for d in record['details']:
                    expected=next(place for obj,place in reversed(d['expected']['events']) if obj==d['expected']['target'])
                    actual=d['output']
                    correct=actual['utf8_valid'] and actual['ended_with_eos'] and actual['text']==expected+'。'
                    independent &= expected==d['expected']['answer'] and correct==d['answer_correct']
        row['independent_label_and_score_check']=independent
        row['interface_tensor_bytes']=sum(t.numel()*t.element_size() for t in payload['state'].values())
        row['replay_coverage_all']=all(training['replay_coverage'].values())
        row['old_tensors_exact']=before==frozen_digest(organism,state)
        row['engineering_passed']=all((exact,row['old_tensors_exact'],row['memory']['passed'],training['old_tensors_exact'],row['replay_coverage_all'],independent))
        metrics=[v for stage in row['stages'].values() for k,v in stage.items() if k!='records']
        metrics += [v for stage in row['full_development'].values() for v in stage.values()]
        metrics += [r['both_correct'] for r in row['second_wording_pairs'].values()]
        row['capability_passed']=min(metrics)>=.95 and row['second_training_wording']['answer_accuracy']>=.95 and row['language_development']['negative_log_likelihood']<=plan['source_development_nll']*plan['retention_ratio_max']
        result['trials'].append(row)
        (out/f'seed{seed}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'seed':seed,'stages':{s:{k:v for k,v in d.items() if k!='records'} for s,d in row['stages'].items()},
            'language_nll':row['language_development']['negative_log_likelihood'],'passed':row['capability_passed']}),flush=True)
        del model,organism,state,payload,restored
    if protected_files(plan)!=protected:raise AssertionError('Protected files changed')
    result['passed']=all(t['engineering_passed'] and t['capability_passed'] for t in result['trials'])
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
