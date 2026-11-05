"""Audit actual P18 exposures, protected lineage and evaluation evidence."""
import json,re
import torch
from chinese_curriculum_common import ROOT,read,sha,plan_data
from spora.curriculum import CoveringSampler
from spora.language_bridge import PAD


def main():
    plan,docs,examples,groups,labels=plan_data()
    root=ROOT/plan['output']
    baseline=read(plan['output']+'/baseline.json')
    complete=read(plan['output']+'/complete.json')
    final=read(plan['output']+'/final-evaluation.json')
    metrics=[json.loads(line) for line in (root/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    protected=read('reports/P17-results.json')['verification']['protected_hashes']
    for path,expected in protected.items():
        if sha(path)!=expected:raise AssertionError('Protected file changed: '+path)
    p17=read('reports/P17-results.json')
    if sha(plan['source_checkpoint'])!=plan['source_sha256']:raise AssertionError('Source interface changed')
    if sha(p17['plan']['initial_interface'])!=p17['baseline']['interface_sha256']:raise AssertionError('Initial interface changed')
    payload=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
    old=torch.load(ROOT/plan['source_checkpoint'],map_location='cpu',weights_only=True)
    sampler=CoveringSampler(groups,plan['schedule'],plan['seed'])
    exposure={'focus':{'presentations':0,'targets':0},'basic':{'presentations':0,'targets':0}}
    for _ in range(payload['steps']):
        for index in sampler.take(plan['batch']).tolist():
            group='focus' if labels[index].startswith('focus-') else 'basic'
            exposure[group]['presentations']+=1
            exposure[group]['targets']+=int((examples[index,1:]!=PAD).sum())
    saved=payload['covering_sampler']
    reconstructed=sampler.snapshot()
    matching=all(reconstructed[k]==saved[k] for k in ('signature','orders','positions','draws','counts')) and torch.equal(reconstructed['rng'],saved['rng'])
    if not matching or sum(v['targets'] for v in exposure.values())!=payload['trained_tokens']:raise AssertionError('Coverage accounting mismatch')
    if not all(metrics[i]['stage_steps']==(i+1)*100 for i in range(len(metrics))):raise AssertionError('Missed check interval')
    if payload['steps']!=complete['new_steps'] or payload['steps']>1000:raise AssertionError('Budget mismatch')
    tests=(ROOT/'runs/p18-regression.log').read_text()
    if not tests.rstrip().endswith('OK'):raise AssertionError('Regression failed')
    old_rows=final['previous_probes']['rows'];new_rows=final['new_probes']['rows']
    level=[row for row in old_rows if row['kind']=='level']
    trivial=sum(not any('\u4e00'<=c<='\u9fff' for c in row['text']) for row in old_rows+new_rows)
    result={'date':'2026-11-05','parent_commit':'a240d42ac19a3f153103a5894eceae6384259a18',
        'plan':plan,'data':read('runs/chinese-curriculum-v2/readiness.json'),'baseline':baseline,
        'periodic_checks':metrics,'completion':complete,'final':final,'exposure':exposure,
        'verification':{'protected_previous_files':len(protected),'source_checkpoint_unchanged':True,
            'pristine_interface_unchanged':True,'sampling_reconstruction_exact':matching,
            'regression_tests':int(re.search(r'Ran (\d+) tests',tests).group(1)),
            'changed_interface_scalar_positions':sum(int((v!=old['state'][k]).sum()) for k,v in payload['state'].items()),
            'old_tensor_checks_passed':all(row['frozen_organism_exact'] for row in metrics),
            'sealed_test_evaluations':1},
        'generation_limits':{'diagnostic_valid_utf8':sum(row['utf8_valid'] for row in old_rows+new_rows),
            'diagnostic_prompts':len(old_rows)+len(new_rows),'correct_content':sum(row['correct'] for row in old_rows+new_rows),
            'level_prompts':len(level),'unique_level_answers':len(set(row['text'] for row in level)),
            'outputs_with_no_han_character':trivial,
            'note':'Valid byte strings and short EOS-terminated output do not establish meaningful words or answers.'},
        'development_relative_reduction':1-final['losses']['development']['negative_log_likelihood']/baseline['development']['negative_log_likelihood'],
        'checkpoint':{'path':str((root/'checkpoint.pt').relative_to(ROOT)),
            'sha256':sha(root/'checkpoint.pt'),'bytes':(root/'checkpoint.pt').stat().st_size,
            'requires_parent':plan['parent'],'requires_curriculum_trainer_for_resume':True},
        'engineering_passed':complete['status']=='complete' and complete['new_steps']==1000 and final['integrity_passed'] and complete['all_training_documents_covered'] and complete['focus_min_presentations']==50,
        'next_training_started':False,
        'interpretation':'Fixed finite synthetic curriculum, no causal control separating data, coverage, optimizer reset or extra compute. Frozen backbone remains a possible limitation, not established cause.'}
    destination=ROOT/'reports/P18-results.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('exposure','verification','generation_limits','development_relative_reduction','checkpoint','engineering_passed')},ensure_ascii=True))
    if not result['engineering_passed']:raise SystemExit(1)


if __name__=='__main__':main()
