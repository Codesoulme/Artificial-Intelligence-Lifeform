"""Assemble P20 failures and final evidence, retaining all raw audit outputs."""
import argparse,json,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',required=True);parser.add_argument('--regression',required=True)
    args=parser.parse_args();audit=json.loads((ROOT/args.audit).read_text('utf-8'))
    log=(ROOT/args.regression).read_text('utf-8-sig');m=re.search(r'Ran (\d+) tests in ([0-9.]+)s',log)
    if not m or not log.strip().endswith('OK'):raise ValueError('Missing successful regression')
    trials=[]
    for path in sorted((ROOT/'runs').glob('p20*-seed*/training.json')):
        row=json.loads(path.read_text('utf-8'));row['directory']=path.parent.relative_to(ROOT).as_posix();trials.append(row)
    interrupted=[]
    for path in sorted((ROOT/'runs').glob('p20*-seed*/interrupted.json')):
        row=json.loads(path.read_text('utf-8'));row['directory']=path.parent.relative_to(ROOT).as_posix();interrupted.append(row)
    for row in trials+interrupted:
        actual=hashlib.sha256((ROOT/row['directory']/'checkpoint.pt').read_bytes()).hexdigest()
        if actual!=row['checkpoint_sha256']:raise AssertionError('Earlier P20 checkpoint changed')
    sources=['spora/language_bridge.py','spora/language_ladder.py','scripts/ladder_task.py',
        'scripts/train_language_ladder.py','scripts/audit_language_ladder.py','scripts/ablate_language_ladder.py','tests/test_language_ladder.py']
    result={'development_trials':trials,'interrupted_trials':interrupted,'final':audit,
        'verification':{'regression_tests':int(m[1]),'regression_seconds':float(m[2]),
            'preserved_P20_checkpoints':len(trials)+len(interrupted),
            'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}},
        'totals':{'updates_across_descendants':sum(t['steps'] for t in trials),
            'interrupted_checkpoint_updates_lower_bound':sum(t['checkpoint_steps'] for t in interrupted),
            'in_memory_mapping_diagnostic_updates':5400,
            'cost_scope':'Completed trials counted exactly, interrupted checkpoints lower bound; uncheckpointed work may be additional. Three 1800-step in-memory diagnostics listed separately.',
            'task_target_presentations':sum(t['task_targets'] for t in trials),
            'replay_target_presentations':sum(t['replay_targets'] for t in trials),
            'loop_seconds_including_checks':sum(t['seconds'] for t in trials)},
        'default_chat_promoted':False,'limitations':['Finite working-context movement tasks, not open Chinese conversation.',
            'All descendants share P18; seeds vary newly initialized language weights and sampling.',
            'Development-guided iterative search; known P19 held-out pair benchmark reused.',
            'Old episodic state is protected, but language-to-autobiographical-memory alignment is not established.',
            'Passed text output checks would not demonstrate consciousness or faithful internal reasoning.']}
    previous=json.loads((ROOT/'reports/P19-results.json').read_text('utf-8'))
    preserved={}
    for t in previous['trials']:
        name=f"runs/p19-{t['arm']}-seed{t['seed']}/checkpoint.pt"
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=t['training']['checkpoint_sha256']:raise AssertionError('P19 checkpoint changed')
        preserved[name]=actual
    result['verification']['preserved_P19_checkpoints']=preserved
    result['verification']['P18_forward_exact_to_P19_implementation']=json.loads((ROOT/'runs/p20-legacy-equivalence.json').read_text('utf-8'))
    earlier=json.loads((ROOT/'configs/chinese-curriculum-1000.json').read_text('utf-8'))
    if hashlib.sha256((ROOT/earlier['source_checkpoint']).read_bytes()).hexdigest()!=earlier['source_sha256']:raise AssertionError('P17 source changed')
    result['verification']['P17_source_unchanged']=True
    development_extra=ROOT/'runs/p20c-full-development-seed7.json'
    if development_extra.exists():
        full=json.loads(development_extra.read_text('utf-8'))
        result['P20c_seed7_full_development_failure']={s:{k:v for k,v in r.items() if k!='records'} for s,r in full.items()}
    diagnostic=ROOT/'runs/p20g-seed11-development-diagnostic.json'
    if diagnostic.exists():
        result['P20g_seed11_development_diagnostic']=json.loads(diagnostic.read_text('utf-8'))
    reference=ROOT/'runs/p20-pointer-source.json'
    if reference.exists():result['method_reference']=json.loads(reference.read_text('utf-8'))
    recency_reference=ROOT/'runs/p20-recency-source.json'
    if recency_reference.exists():result['recency_reference']=json.loads(recency_reference.read_text('utf-8'))
    ablation=ROOT/'runs/p20h-development-ablation.json'
    if ablation.exists():result['inference_interventions']=json.loads(ablation.read_text('utf-8'))
    final_ablation=ROOT/'runs/p20j-development-ablation.json'
    if final_ablation.exists():result['final_inference_interventions']=json.loads(final_ablation.read_text('utf-8'))
    result['additional_development_diagnostics']={
        p.name:json.loads(p.read_text('utf-8'))
        for pattern in ('p20h-seed23-diagnostic-*.json','p20h-query-window-diagnostic-*.json')
        for p in sorted((ROOT/'runs').glob(pattern))}
    result['final_trial_parameters']=[{'seed':t['seed'],'trainable_parameters':t['training']['trainable_parameters'],
        'steps':t['training']['steps'],'interface_tensor_bytes':t['interface_tensor_bytes'],
        'checkpoint_bytes':t['checkpoint_bytes'],'latency':t['latency']} for t in audit['trials']]
    out=ROOT/'reports/P20-results.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'totals':result['totals'],'verification':result['verification'],
        'final_passed':audit['passed'],'trials':[{'seed':t['seed'],'stages':{s:{k:v for k,v in r.items() if k!='records'} for s,r in t['stages'].items()},
        'language_nll':t['language_development']['negative_log_likelihood'],'new_wording':t['new_wording']['answer_accuracy'],
        'values_correct':t['values_probes']['correct']} for t in audit['trials']]},indent=2))


if __name__=='__main__':main()
