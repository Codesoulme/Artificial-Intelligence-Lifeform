"""Sequential fresh evaluation after a development selection has been sealed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    selection=Path('runs/p3-development/development.json')
    if not selection.exists():raise RuntimeError('Development must finish first')
    digest=hashlib.sha256(selection.read_bytes()).hexdigest()
    data=json.loads(selection.read_text(encoding='utf-8'))
    if hashlib.sha256(Path(data['checkpoint']).read_bytes()).hexdigest()!=data['sha256']:
        raise ValueError('Development foundation changed')
    plan=json.loads(Path('configs/p3-plan.json').read_text())
    locked=Path('runs/p3-selection-lock.json')
    if locked.exists():raise FileExistsError('Preserve locked audit')
    locked.write_text(json.dumps({'selection_sha256':digest,'selected':data['selected'],
                                 'plan_sha256':hashlib.sha256(Path('configs/p3-plan.json').read_bytes()).hexdigest(),
                                 'audit_seed':plan['audit_seed'],'seeds':plan['foundation_seeds']},indent=2),encoding='utf-8')
    for seed in plan['foundation_seeds']:
        target=f'runs/p3-audit-seed{seed}'
        subprocess.run([sys.executable,'-m','spora.consolidate_p3','--phase','audit',
                        '--checkpoint',f'checkpoints/p2-v1-delta-seed{seed}.pt',
                        '--output-dir',target,'--selection',str(selection),'--device','cuda'],check=True)
        if hashlib.sha256(selection.read_bytes()).hexdigest()!=digest:raise AssertionError('Selection changed')
        subprocess.run([sys.executable,'-m','scripts.plasticity_p3','--audit-dir',target,
                        '--output',f'runs/p3-plasticity-seed{seed}.json'],check=True)
    print('Three fixed-foundation audits complete',flush=True)


if __name__=='__main__':main()
