"""Verify the exact authorized budget and summarize P17 without more updates."""
import hashlib,json,math,re,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_prechinese_runtime import protected_all
from spora.text_data import read_manifest,windows
from spora.language_bridge import PAD


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    plan=read('configs/chinese-pilot-100-values.json')
    baseline,final=read('runs/p17-baseline.json'),read('runs/p17-final.json')
    completion=read(plan['output']+'/complete.json')
    initial=torch.load(ROOT/plan['initial_interface'],weights_only=True,map_location='cpu')
    checkpoint=ROOT/plan['output']/'checkpoint.pt'
    trained=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if trained['steps']!=100 or completion['steps']!=100:raise AssertionError('Not exactly 100 steps')
    if not all(completion[k] for k in ('parent_unchanged','all_organism_tensors_unchanged')):raise AssertionError('Original neural state changed')
    protected=protected_all()
    for name,info in read('reports/P14-P16-results.json')['candidate']['files'].items():
        if digest(ROOT/name)!=info['sha256']:raise AssertionError('P16 individual changed')
        protected[name]=info['sha256']
    if digest(ROOT/plan['initial_interface'])!=baseline['interface_sha256']:raise AssertionError('Initial interface changed')
    documents=read_manifest(ROOT/plan['manifest'])
    examples=windows(documents,'train',128)
    labels=[]
    for doc in documents:
        if doc['split']=='train':labels += [doc['path']]*len(windows([doc],'train',128))
    generator=torch.Generator().manual_seed(plan['seed'])
    counts={doc['path']:0 for doc in documents if doc['split']=='train'}
    targets=focus_targets=focus_presentations=0
    for _ in range(100):
        indices=torch.randint(len(examples),(4,),generator=generator)
        for index in indices.tolist():
            name=labels[index];counts[name]+=1
            count=int((examples[index,1:]!=PAD).sum());targets+=count
            if name.startswith('focus-'):focus_targets+=count;focus_presentations+=1
    if targets!=trained['trained_tokens'] or not torch.equal(generator.get_state(),trained['sampler_rng']):
        raise AssertionError('Actual sampler cannot be reconstructed from fixed protocol')
    changed=[k for k,v in initial['state'].items() if not torch.equal(v,trained['state'][k])]
    metrics=[json.loads(line) for line in (ROOT/plan['output']/'metrics.jsonl').read_text().splitlines()]
    finite=all(math.isfinite(row[key]) for row in metrics for key in ('train_loss','development_loss'))
    checks=(ROOT/'runs/p17-preflight-tests.log').read_text()
    if not checks.rstrip().endswith('OK'):raise AssertionError('Relevant tests failed')
    development_before=baseline['losses']['development']['negative_log_likelihood']
    development_after=final['losses']['development']['negative_log_likelihood']
    report={'date':'2026-11-05','plan':plan,'data':read('runs/chinese-pilot-values-v1/readiness.json'),
        'baseline':baseline,'final':final,'completion':completion,'metrics':metrics,
        'exposure':{'total_window_presentations':400,'focus_window_presentations':focus_presentations,
            'actual_focus_window_fraction':focus_presentations/400,'targets':targets,'focus_targets':focus_targets,
            'actual_focus_target_fraction':focus_targets/targets,'per_training_document_window_presentations':counts},
        'verification':{'protected_individual_files':len(protected),'protected_hashes':protected,
            'initial_interface_unchanged':True,'changed_interface_tensors':len(changed),
            'changed_interface_scalar_positions':sum(int((v!=trained['state'][k]).sum()) for k,v in initial['state'].items()),
            'sampler_and_target_counts_exact':True,'finite_losses':finite,
            'relevant_tests':int(re.search(r'Ran (\d+) tests',checks).group(1)),
            'final_evaluation_in_separate_process':True,'sealed_test_evaluations':1},
        'development_nll_relative_reduction':1-development_after/development_before,
        'checkpoint':{'path':str(checkpoint.relative_to(ROOT)),'sha256':digest(checkpoint),'bytes':checkpoint.stat().st_size,
            'format':trained['format'],'requires_parent':plan['parent']},
        'engineering_passed':bool(changed) and finite and final['measurements_valid'] and baseline['measurements_valid'] and development_after<development_before,
        'content_probe_correct':sum(p['correct'] for p in final['probes']),
        'content_probe_total':len(final['probes']),
        'next_training_started':False,
        'interpretation':'One mixed-data 100-step run; no no-focus control, no claim of causally isolated curriculum benefit or stable values. Teacher-forced likelihood and autonomous generation measured separately.'}
    destination=ROOT/'reports/P17-results.json'
    with destination.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k in ('exposure','verification','development_nll_relative_reduction','checkpoint','engineering_passed','content_probe_correct','content_probe_total')},ensure_ascii=False))
    if not report['engineering_passed']:raise SystemExit(1)


if __name__=='__main__':main()
