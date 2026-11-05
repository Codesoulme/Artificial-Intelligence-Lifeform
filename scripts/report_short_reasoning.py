"""Publish complete measurements and integrity evidence, without model weights."""
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    report=json.loads((ROOT/'runs/p19-final-audit/audit.json').read_text(encoding='utf-8'))
    log=(ROOT/'runs/p19-regression.log').read_text(encoding='utf-8-sig')
    match=re.search(r'Ran (\d+) tests in ([0-9.]+)s',log)
    if not match or not log.strip().endswith('OK'):raise ValueError('Regression not successful')
    prior=json.loads((ROOT/'configs/chinese-curriculum-1000.json').read_text(encoding='utf-8'))
    if sha(prior['source_checkpoint'])!=prior['source_sha256']:raise AssertionError('P17 interface changed')
    files=['spora/language_bridge.py','spora/short_reasoning.py','scripts/reasoning_task.py',
        'scripts/train_short_reasoning.py','scripts/audit_short_reasoning.py','tests/test_short_reasoning.py']
    report['verification']={'regression_tests':int(match[1]),'regression_seconds':float(match[2]),
        'regression_log_sha256':sha('runs/p19-regression.log'),
        'P17_source_unchanged':True,'P17_source_path':prior['source_checkpoint'],'P17_source_sha256':prior['source_sha256'],
        'source_code_sha256':{p:sha(p) for p in files},
        'implementation_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    summaries={}
    for arm in report['plan']['arms']:
        trials=[t for t in report['trials'] if t['arm']==arm]
        summaries[arm]={'seeds':[t['seed'] for t in trials],
            'answers':[t['standard']['answer_accuracy'] for t in trials],
            'trace':[t['standard'].get('trace_correct') for t in trials],
            'paired_query':[t['paired_query']['both_answers_correct'] for t in trials],
            'paired_reversal_changed':[t['paired_reversal_changed']['both_answers_correct'] for t in trials],
            'new_wording':[t['new_wording']['answer_accuracy'] for t in trials],
            'training_partition':[t['seen_training']['answer_accuracy'] for t in trials],
            'old_language_development_nll':[t['retention']['development']['negative_log_likelihood'] for t in trials],
            'focus_training_nll':[t['retention']['focus_training']['negative_log_likelihood'] for t in trials],
            'values_probe_correct':[t['retention']['probes']['correct'] for t in trials],
            'median_trial_train_seconds':statistics.median(t['training']['train_seconds'] for t in trials),
            'latency_seconds':[t['latency']['median_seconds'] for t in trials],
            'all_capability_gates_passed':all(t['capability_passed'] for t in trials)}
    report['summary']=summaries
    report['totals']={'independent_trial_updates':sum(t['training']['steps'] for t in report['trials']),
        'task_target_presentations':sum(t['training']['task_target_presentations'] for t in report['trials']),
        'replay_target_presentations':sum(t['training']['replay_target_presentations'] for t in report['trials']),
        'optimization_seconds':sum(t['training']['train_seconds'] for t in report['trials']),
        'training_loop_seconds':sum(t['training']['seconds_including_evaluation'] for t in report['trials'])}
    destination=ROOT/'reports/P19-results.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summaries':summaries,'totals':report['totals'],'baseline':report['baselines'],
        'source_development':report['source_retention']['development'],
        'source_focus':report['source_retention']['focus_training'],'bytes':destination.stat().st_size},indent=2))


if __name__=='__main__':main()
