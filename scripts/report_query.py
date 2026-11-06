"""Collect every completed P22 experiment, including failed recipes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from chinese_curriculum_common import ROOT,sha

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tags',nargs='+',default=['p22a','p22b'])
    args=parser.parse_args();out=ROOT/'reports/P22-results.json'
    if out.exists():raise FileExistsError(out)
    audits={};experiments=[]
    for tag in args.tags:
        audits[tag]=json.loads((ROOT/f'runs/{tag}-audit/audit.json').read_text('utf-8'))
        for path in sorted((ROOT/'runs').glob(f'{tag}-*-seed*/training.json')):
            row=json.loads(path.read_text('utf-8'))
            row['folder']=str(path.parent.relative_to(ROOT)).replace('\\','/')
            row['configuration']=json.loads((path.parent/'config.json').read_text('utf-8'))
            if sha(path.parent/'checkpoint.pt')!=row['checkpoint_sha256']:raise ValueError('Candidate changed')
            for name,digest in row['configuration']['protected'].items():
                if sha(name)!=digest:raise ValueError('Protected file changed')
            experiments.append(row)
    cost={key:sum(row[key] for row in experiments) for key in ('steps','task_targets','replay_targets','seconds')}
    cost['scope']='Sum across separate models, excludes P21 ancestry, baseline and independent audit; timing may include concurrent audit contention, not an arm speed comparison. Targets count UTF-8 bytes/EOS presentations.'
    log=ROOT/'runs/p22-regression.log';raw=log.read_bytes()
    result=dict(audits=audits,experiments=experiments,cost=cost,
        source_sha256={name:sha(name) for name in ('scripts/report_query.py','scripts/audit_query.py','spora/query_bridge.py','spora/anchor_query.py','scripts/anchor_labels.py','scripts/query_task.py','tests/test_query_bridge.py','tests/test_anchor_query.py')},
        audit_sha256={tag:sha(f'runs/{tag}-audit/audit.json') for tag in args.tags},
        regression_log=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8'),
        startup_failure='One unittest import failure fixed before any training; no checkpoint produced or overwritten.')
    mechanism=ROOT/'runs/p22c-mechanism.json'
    if mechanism.exists():
        result['mechanism']=json.loads(mechanism.read_text('utf-8'))
        result['source_sha256']['scripts/diagnose_query.py']=sha('scripts/diagnose_query.py')
        result['mechanism_sha256']=sha(mechanism)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(cost=cost,bytes=out.stat().st_size,latest_passed=audits[args.tags[-1]]['passed'])))

if __name__=='__main__':main()
