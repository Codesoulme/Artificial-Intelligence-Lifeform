"""Finite sequential GPU jobs; skip only already completed runs, never overwrite."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parents[1]
    for kind in ('split','shared'):
        for seed in (7,11,23):
            name=f'p4-{kind}-seed{seed}';run=root/'runs'/name
            checkpoint=root/'checkpoints'/f'{name}.pt'
            if (run/'summary.json').exists():
                if not checkpoint.exists():raise FileNotFoundError(checkpoint)
                print(json.dumps({'already_complete':name}),flush=True);continue
            if run.exists() or checkpoint.exists():raise FileExistsError('Incomplete run requires explicit review: '+name)
            print(json.dumps({'starting_training':name}),flush=True)
            log=root/'runs'/f'{name}.console.log'
            with log.open('x',encoding='utf-8') as output:
                subprocess.run([sys.executable,'-m','spora.train_p4','--kind',kind,'--seed',str(seed),
                    '--run-dir',str(run),'--checkpoint',str(checkpoint)],cwd=root,stdout=output,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((run/'summary.json').read_text(encoding='utf-8'))
            print(json.dumps({'training_complete':name,'seconds':summary['elapsed_seconds'],
                              'development32':summary['development']['32']}),flush=True)
    # All training choices finish before the held-out audit is opened.
    for kind in ('split','shared'):
        for seed in (7,11,23):
            name=f'p4-{kind}-seed{seed}';directory=root/'runs'/f'{name}-audit'
            if directory.exists():raise FileExistsError(directory)
            print(json.dumps({'starting_audit':name}),flush=True)
            with (root/'runs'/f'{name}-audit.console.log').open('x',encoding='utf-8') as output:
                subprocess.run([sys.executable,'-m','spora.audit_p4','--checkpoint',str(root/'checkpoints'/f'{name}.pt'),
                    '--output-dir',str(directory)],cwd=root,stdout=output,stderr=subprocess.STDOUT,check=True)
            result=json.loads((directory/'audit.json').read_text(encoding='utf-8'))
            print(json.dumps({'audit_complete':name,'gate':result['gate'],'seconds':result['seconds']}),flush=True)
    print(json.dumps({'completed':'six fixed P4 models and audits'}),flush=True)


if __name__=='__main__':main()
