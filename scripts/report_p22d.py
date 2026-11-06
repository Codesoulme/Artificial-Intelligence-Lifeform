"""Archive D continuation and its interrupted startup without changing P22 history."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'reports/P22d-results.json'
    if out.exists():raise FileExistsError(out)
    result=json.loads((ROOT/'runs/p22d2-audit/audit.json').read_text('utf-8'))
    interruption=json.loads((ROOT/'runs/p22d-context-seed7/interrupted.json').read_text('utf-8'))
    if hashlib.file_digest((ROOT/'runs/p22d-context-seed7/checkpoint.pt').open('rb'),'sha256').hexdigest()!=interruption['checkpoint_sha256']:raise ValueError('Interrupted checkpoint changed')
    result['interruption']=interruption
    result['updates_lower_bound']=sum(row['training']['steps'] for row in result['trials'])+interruption['saved_steps']
    result['source_code_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('scripts/train_query_triplets.py','scripts/audit_query_triplets.py','scripts/report_p22d.py')}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],final_test_opened=result['final_test_opened'],updates_lower_bound=result['updates_lower_bound'])))
if __name__=='__main__':main()
