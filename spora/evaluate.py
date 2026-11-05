"""One-shot audit of held-out worlds and a fresh-process restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from .checkpoint import load_checkpoint, save_checkpoint
from .tasks import mappings, absorb


def bootstrap_interval(values, seed=95631):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),size=(3000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':[float(x) for x in np.quantile(samples,[.025,.975])]}


def isolated_query(checkpoint, keys, reset_fast=False):
    command=[sys.executable,'-m','spora.infer','--checkpoint',str(Path(checkpoint).resolve())]
    if reset_fast: command.append('--reset-fast')
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    environment['PYTHONUTF8']='1'
    with tempfile.TemporaryDirectory(prefix='spora-query-') as directory:
        result=subprocess.run(command,input=json.dumps({'keys':keys})+'\n',text=True,
                              capture_output=True,cwd=directory,env=environment,timeout=90)
    if result.returncode:
        raise RuntimeError('Restart worker failed: '+result.stderr)
    return json.loads(result.stdout.strip())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--worlds',type=int,default=128)
    parser.add_argument('--pairs',type=int,default=8)
    parser.add_argument('--audit-seed',type=int,default=200000)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    output=Path(args.output_dir)
    if (output/'audit.json').exists():
        raise FileExistsError('Audit already exists. Do not silently overwrite an observed test.')
    if args.worlds < 2:
        raise ValueError('At least two independent worlds required')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model,_,_,data=load_checkpoint(args.checkpoint,args.device)
    generator=torch.Generator().manual_seed(args.audit_seed)
    results={name:[] for name in ['written','no_write','after_new_old','after_new_new','alternate_mapping']}
    start=time.perf_counter()
    first=None
    for offset in range(0,args.worlds,8):
        batch=min(8,args.worlds-offset)
        keys,values=mappings(generator,batch,2*args.pairs,model.config,args.device)
        old_keys,old_values=keys[:,:args.pairs],values[:,:args.pairs]
        initial,_=model.initial_state(batch)
        fast,velocity=absorb(model,old_keys,old_values)
        with torch.no_grad():
            predictions=model.predict_many(old_keys,fast).argmax(-1)
            baseline=model.predict_many(old_keys,initial).argmax(-1)
        results['written'].extend((predictions==old_values).float().mean(1).tolist())
        results['no_write'].extend((baseline==old_values).float().mean(1).tolist())
        alternate_values=(old_values+1)%model.config.value_count
        alternative,_=absorb(model,old_keys,alternate_values)
        with torch.no_grad():
            changed=model.predict_many(old_keys,alternative).argmax(-1)
        results['alternate_mapping'].extend((changed==alternate_values).float().mean(1).tolist())
        if first is None:
            first={'keys':old_keys[0].cpu().tolist(),'targets':old_values[0].cpu().tolist(),
                   'predictions':predictions[0].cpu().tolist(),
                   'fast':[[p[:1].clone() for p in site] for site in fast],
                   'velocity':[[p[:1].clone() for p in site] for site in velocity]}
        continued,_=absorb(model,keys[:,args.pairs:],values[:,args.pairs:],fast,velocity)
        with torch.no_grad():
            old=model.predict_many(old_keys,continued).argmax(-1)
            new=model.predict_many(keys[:,args.pairs:],continued).argmax(-1)
        results['after_new_old'].extend((old==old_values).float().mean(1).tolist())
        results['after_new_new'].extend((new==values[:,args.pairs:]).float().mean(1).tolist())
    individual=output/'individual.pt'
    save_checkpoint(individual,model,first.pop('fast'),first.pop('velocity'),cursor=args.pairs,
                    metadata={'stage':'P1 audited individual','parent_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()})
    restarted=isolated_query(individual,first['keys'])
    reset=isolated_query(individual,first['keys'],True)
    match=restarted['predictions']==first['predictions']
    if not match:
        raise AssertionError('Predictions differ across restart')
    summaries={name:bootstrap_interval(values) for name,values in results.items()}
    gain=bootstrap_interval(np.array(results['written'])-np.array(results['no_write']))
    loss=bootstrap_interval(np.array(results['written'])-np.array(results['after_new_old']))
    report={
        'foundation':str(Path(args.checkpoint).resolve()),'foundation_training':data['metadata'],
        'parameters':model.parameter_count(),'worlds':args.worlds,'facts_per_world':args.pairs,
        'audit_seed':args.audit_seed,'chance_accuracy':1/model.config.value_count,
        'scores':summaries,'paired_memory_gain':gain,'paired_old_task_loss':loss,
        'restart':{'same_predictions':match,'keys':first['keys'],'targets':first['targets'],
                   'in_process':first['predictions'],'new_process':restarted['predictions'],
                   'fast_reset':reset['predictions'],'individual':str(individual.resolve()),
                   'input_contract':'worker receives checkpoint and current keys; never targets or past events',
                   'isolation':'new process, empty cwd, post-load Python file/network audit guard; not an OS sandbox'},
        'p1_gain_gate':gain['ci95'][0]>.10,
        'scope':'Within-seed exploratory interval across worlds; not a claim of cross-training-seed significance.',
        'limitations':['No slow consolidation evidence','No language ability tested',
                      'No equal-compute trained baseline yet','No life or consciousness claim'],
        'elapsed_seconds':time.perf_counter()-start,
    }
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'world_scores.json').write_text(json.dumps(results),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
