"""Complete orthogonal-current revision; stop if any preregistered gate fails."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(command,name):
    with (ROOT/'runs'/f'{name}.console.log').open('x',encoding='utf-8') as log:
        subprocess.run([sys.executable]+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    for seed in (7,11,23):
        name=f'p5b-orthogonal90-seed{seed}'
        if seed!=7 and not (ROOT/'runs'/name/'summary.json').exists():
            print(json.dumps({'training':name}),flush=True)
            run(['scripts/run_p5.py','--variant','orthogonal90','--seed',str(seed)],name)
        if not (ROOT/'runs'/name/'summary.json').exists():raise RuntimeError('Incomplete seed7 development')
    for seed in (7,11,23):
        name=f'p5b-orthogonal90-seed{seed}';print(json.dumps({'audit':name}),flush=True)
        if not (ROOT/'runs'/f'{name}-audit/audit.json').exists():
            run(['scripts/audit_p5.py','--checkpoint',f'checkpoints/{name}.pt','--output',f'runs/{name}-audit','--seed','1610000'],name+'-audit')
        r=json.loads((ROOT/'runs'/f'{name}-audit/audit.json').read_text())
        if not r['gate']['passed'] or any(x['accuracy']<.95 for x in r['varied_entities64'].values()):
            print(json.dumps({'revision_needed':name,'gate':r['gate'],'varied':r['varied_entities64']}),flush=True);return
        print(json.dumps({'audit_passed':name}),flush=True)
    for seed in (7,11,23):
        for entities in (4,64):
            name=f'p6-seed{seed}-entities{entities}';print(json.dumps({'growth_audit':name}),flush=True)
            if not (ROOT/'runs'/name/'audit.json').exists():
                run(['scripts/audit_growth.py','--checkpoint',f'checkpoints/p5b-orthogonal90-seed{seed}.pt','--output',f'runs/{name}',
                     '--worlds','8','--entities',str(entities),'--seed','1810000'],name)
            r=json.loads((ROOT/'runs'/name/'audit.json').read_text())
            if not r['passed']:
                print(json.dumps({'growth_revision_needed':name}),flush=True);return
            print(json.dumps({'growth_passed':name}),flush=True)
    print(json.dumps({'completed':'P5b/P6 fixed gates'}),flush=True)

if __name__=='__main__':main()
