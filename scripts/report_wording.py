"""Package P21 measurements without discarding failures or raw generations."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.chinese_curriculum_common import ROOT, sha

def main():
    source=ROOT/'runs/p21a-audit/audit.json'
    result=json.loads(source.read_text('utf-8'))
    destination=ROOT/'reports/P21-results.json'
    if destination.exists():raise FileExistsError(destination)
    for row in result['trials']:
        seed=row['seed'];path=ROOT/f'runs/p21a-seed{seed}'
        row['baseline']=json.loads((path/'baseline.json').read_text('utf-8'))
        row['configuration']=json.loads((path/'config.json').read_text('utf-8'))
        row['checkpoint_bytes']=(path/'checkpoint.pt').stat().st_size
    result['provenance']={name:sha(name) for name in (
        'scripts/train_wording.py','scripts/wording_task.py','scripts/audit_wording.py',
        'scripts/report_wording.py','configs/p21-wording.json','tests/test_wording.py',
        'runs/p21a-audit/audit.json','runs/p21-regression.log')}
    result['cost']={key:sum(row['training'][key] for row in result['trials'])
                    for key in ('steps','task_targets','replay_targets','seconds')}
    result['cost']['scope']='Three separate continuations; excludes inherited P20 training and baseline/final-audit wall time. Target counts are UTF-8 bytes/EOS presentations, not unique tokens.'
    result['startup_failure']='One direct-script import failure occurred before run directories or training updates; fixed repository import path, no checkpoint lost.'
    result['regression_log']=(ROOT/'runs/p21-regression.log').read_text('utf-16') if (ROOT/'runs/p21-regression.log').read_bytes().startswith(b'\xff\xfe') else (ROOT/'runs/p21-regression.log').read_text('utf-8')
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],cost=result['cost'],bytes=destination.stat().st_size,protected_files=result['protected_files'])))

if __name__=='__main__':main()
