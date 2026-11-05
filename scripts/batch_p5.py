"""Complete fixed P5 training jobs, lock selection, then open held-out streams."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(command,name):
    with (ROOT/'runs'/f'{name}.console.log').open('x',encoding='utf-8') as log:
        subprocess.run([sys.executable]+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    first=ROOT/'runs/p5-rls90-seed7/summary.json'
    if not first.exists():raise RuntimeError('Finish initial developer run before batch')
    for variant,seed in [('delta110',7),('rls90',11),('rls90',23)]:
        name=f'p5-{variant}-seed{seed}';print(json.dumps({'training':name}),flush=True)
        run(['scripts/run_p5.py','--variant',variant,'--seed',str(seed)],name)
        summary=json.loads((ROOT/'runs'/name/'summary.json').read_text())
        print(json.dumps({'completed':name,'development64':summary['development']['64']}),flush=True)
    candidates={name:json.loads((ROOT/'runs'/f'p5-{name}-seed7/summary.json').read_text()) for name in ('rls90','delta110')}
    def minimum(summary):
        return min(r[m]['accuracy'] for r in summary['development'].values() for m in ('current','history','invalid_pair_unknown','future_unknown') if r[m]['accuracy'] is not None)
    selected=max(candidates,key=lambda n:minimum(candidates[n]))
    selection={'selected':selected,'development_minimum':{n:minimum(s) for n,s in candidates.items()},'audit_opened':False}
    path=ROOT/'configs/p5-selection.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(selection,indent=2))
    if selected!='rls90' or minimum(candidates[selected])<.95:
        print(json.dumps({'needs_iteration':selection}),flush=True);return
    for seed in (7,11,23):
        name=f'p5-rls90-seed{seed}';print(json.dumps({'audit':name}),flush=True)
        run(['scripts/audit_p5.py','--checkpoint',f'checkpoints/{name}.pt','--output',f'runs/{name}-audit'],name+'-audit')
        result=json.loads((ROOT/'runs'/f'{name}-audit/audit.json').read_text())
        print(json.dumps({'audit_completed':name,'gate':result['gate'],'varied_entities':result['varied_entities64']}),flush=True)

if __name__=='__main__':main()
