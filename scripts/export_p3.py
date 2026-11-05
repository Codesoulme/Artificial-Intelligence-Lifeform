"""Export only the fixed seed7/world0 candidate after all audits finish."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
from spora.checkpoint import load_checkpoint
from spora.evaluate import isolated_query
from spora.tasks import mappings


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    destination=Path('checkpoints/individual-p3-candidate.pt')
    manifest_path=Path('reports/P3-candidate.json')
    if destination.exists() or manifest_path.exists():raise FileExistsError('Never replace an existing individual')
    audits=[read(f'runs/p3-audit-seed{s}/audit.json') for s in (7,11,23)]
    for audit in audits:
        if not audit['worlds']:raise ValueError('Incomplete audit')
    chosen=audits[0]['worlds'][0]
    if len(chosen['stages'])!=3 or not all(r['gate'] for r in chosen['stages']):
        raise ValueError('Fixed candidate failed; do not choose a nicer world')
    stress=read('runs/p3-plasticity-seed7.json')['results'][0]
    if stress.get('correction_success',0)<.95 or stress.get('final_all_known',0)<.95:
        raise ValueError('Fixed candidate failed sequential correction')
    source=Path('runs/p3-audit-seed7/world0-stage3-candidate.pt')
    torch.set_num_threads(2)
    model,fast,velocity,data=load_checkpoint(source)
    keys,values=mappings(torch.Generator().manual_seed(910000),1,24,model.config,'cpu')
    with torch.no_grad():expected=model.predict_many(keys,fast).argmax(-1).flatten().tolist()
    if expected!=values.flatten().tolist():raise AssertionError('Fixed candidate facts changed')
    with source.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
    resumed=isolated_query(destination,keys.flatten().tolist(),True)['predictions']
    if resumed!=expected:raise AssertionError('Export restart changed answers')
    result={'source':str(source.resolve()),'candidate':str(destination.resolve()),
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'foundation_seed':7,'world':0,'facts':24,'cursor':data['cursor'],
            'residual_write':model.config.residual_write,'restart_match':True,
            'evaluation_manifest_only':True,'queries':keys.flatten().tolist(),
            'expected':values.flatten().tolist(),
            'limits':['Independent research branch; not default individual migration',
                      'This manifest is test evidence and is never read by model inference',
                      'Future writes can still interfere; do not claim unlimited retention']}
    manifest_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
