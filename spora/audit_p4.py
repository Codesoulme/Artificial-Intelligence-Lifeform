"""One locked audit, matched random worlds, separate validity groups and controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch
from .checkpoint import load_checkpoint,save_checkpoint
from .p4 import evaluate,consume
from .temporal_tasks import events


def isolated_query(checkpoint,keys,modes,times,reset=False):
    command=[sys.executable,'-m','spora.temporal_infer','--checkpoint',str(Path(checkpoint).resolve())]
    if reset:command.append('--reset-fast')
    environment=os.environ.copy();environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    environment['PYTHONUTF8']='1'
    with tempfile.TemporaryDirectory(prefix='spora-temporal-query-') as directory:
        result=subprocess.run(command,input=json.dumps({'keys':keys,'modes':modes,'times':times})+'\n',
            text=True,capture_output=True,cwd=directory,env=environment,timeout=90)
    if result.returncode:raise RuntimeError('Temporal restart failed: '+result.stderr)
    return json.loads(result.stdout.strip())


@torch.no_grad()
def immediate(model,worlds=64,seed=1110000):
    device=next(model.parameters()).device;generator=torch.Generator().manual_seed(seed+300)
    results={name:{'correct':0,'total':0} for name in ('correction','untrusted_preserves_prediction','untrusted_current','repeat','second_correction')}
    for start in range(0,worlds,4):
        record=events(generator,min(4,worlds-start),8,model.config,device)
        fast,velocity=model.initial_state(record['keys'].shape[0]);before=None
        for i in range(8):
            fast,velocity=model.write_temporal(fast,velocity,record['keys'][:,i],record['values'][:,i],
                record['times'][:,i],record['trusted'][:,i])
            if i<4:continue
            answer=model(record['keys'][:,0],fast).argmax(-1)
            if i==5:
                parts={'untrusted_preserves_prediction':answer==before,'untrusted_current':answer==record['values'][:,4]}
            else:
                name={4:'correction',6:'repeat',7:'second_correction'}[i]
                parts={name:answer==record['values'][:,i]}
            for name,values in parts.items():
                results[name]['correct']+=int(values.sum());results[name]['total']+=values.numel()
            before=answer.clone()
    return {k:{**v,'accuracy':v['correct']/v['total']} for k,v in results.items()}


def restart_check(foundation,directory):
    previous_threads=torch.get_num_threads();torch.set_num_threads(2)
    model,_,_,_=load_checkpoint(foundation)
    record=events(torch.Generator().manual_seed(1110900),1,8,model.config,'cpu',offset=0)
    fast,velocity=consume(model,record)
    keys=[int(record['keys'][0,0])]*2+record['keys'][0].tolist()+[(int(record['keys'][0,0])+1)%64]
    modes=[0,1]+[1]*8+[1];times=[0,8]+list(range(8))+[0]
    tensors=[torch.tensor(x).reshape(1,-1) for x in (keys,modes,times)]
    with torch.no_grad():
        original=model.predict_many(tensors[0],fast,tensors[1],tensors[2]).softmax(-1).squeeze(0).tolist()
    sample=directory/'restart-individual.pt'
    save_checkpoint(sample,model,fast,velocity,cursor=8,metadata={'stage':'P4 restart test only'})
    cold=isolated_query(sample,keys,modes,times)
    reset=isolated_query(sample,keys,modes,times,True)
    expected=[int(record['values'][0,7]),model.config.unknown]+record['values'][0].tolist()+[model.config.unknown]
    # Continue from the reloaded file, compare all tensors to uninterrupted writing.
    loaded,ff,vv,data=load_checkpoint(sample)
    argument=(torch.tensor([int(record['keys'][0,0])]),torch.tensor([2]),torch.tensor([8]),torch.ones(1))
    uninterrupted,uv=model.write_temporal(fast,velocity,*argument)
    continued,cv=loaded.write_temporal(ff,vv,*argument)
    continuation_exact=all(torch.equal(a,b) for aa,bb in zip(uninterrupted,continued) for a,b in zip(aa,bb))
    torch.set_num_threads(previous_threads)
    return {'identical_probabilities':original==cold['probabilities'],
            'continuation_exact_tensors':continuation_exact,'queries':{'keys':keys,'modes':modes,'times':times},
            'expected':expected,'predictions':cold['predictions'],'reset_predictions':reset['predictions'],
            'guard':'empty cwd; Python audit hook blocks file/network access after loading; not OS sandbox'}


@torch.no_grad()
def early_sequence(model,worlds=64,seed=1111300):
    """Diagnostic outside main gate: untrusted-only key, then two trusted corrections."""
    device=next(model.parameters()).device;generator=torch.Generator().manual_seed(seed)
    result={name:{'correct':0,'total':0} for name in
        ('untrusted_only_current_unknown','untrusted_only_history','first_trusted_current',
         'second_trusted_current','all_three_history','future_unknown')}
    for start in range(0,worlds,4):
        batch=min(4,worlds-start)
        key=torch.randint(model.config.key_count,(batch,),generator=generator).to(device)
        first=torch.randint(model.config.unknown,(batch,),generator=generator).to(device)
        values=[first,(first+1)%model.config.unknown,(first+5)%model.config.unknown]
        fast,velocity=model.initial_state(batch)
        for time in range(3):
            stamp=torch.full_like(key,time);trust=torch.full((batch,),float(time>0),device=device)
            fast,velocity=model.write_temporal(fast,velocity,key,values[time],stamp,trust)
            current=model(key,fast).argmax(-1)
            target=torch.full_like(key,model.config.unknown) if time==0 else values[time]
            name=('untrusted_only_current_unknown','first_trusted_current','second_trusted_current')[time]
            result[name]['correct']+=int((current==target).sum());result[name]['total']+=batch
            if time==0:
                observed=model(key,fast,torch.ones_like(key),stamp).argmax(-1)
                result['untrusted_only_history']['correct']+=int((observed==first).sum())
                result['untrusted_only_history']['total']+=batch
        for time in range(4):
            actual=model(key,fast,torch.ones_like(key),torch.full_like(key,time)).argmax(-1)
            target=values[time] if time<3 else torch.full_like(key,model.config.unknown)
            name='all_three_history' if time<3 else 'future_unknown'
            result[name]['correct']+=int((actual==target).sum());result[name]['total']+=batch
    return {k:{**v,'accuracy':v['correct']/v['total']} for k,v in result.items()}


@torch.no_grad()
def unseen_pairs(model,worlds=64,seed=1111500):
    """Negative pairs beyond the +1 key offset and next-step future used in training."""
    device=next(model.parameters()).device;generator=torch.Generator().manual_seed(seed)
    result={name:{'correct':0,'total':0} for name in ('distant_wrong_key_unknown','unobserved_past_unknown')}
    for start in range(0,worlds,4):
        record=events(generator,min(4,worlds-start),16,model.config,device)
        fast,_=consume(model,record)
        keys=torch.cat([(record['keys']+offset)%model.config.key_count for offset in (7,19,37)],1)
        times=record['times'].repeat(1,3)
        observed=model.predict_many(keys,fast,torch.ones_like(keys),times).argmax(-1)
        result['distant_wrong_key_unknown']['correct']+=int((observed==model.config.unknown).sum())
        result['distant_wrong_key_unknown']['total']+=observed.numel()
        mask=record['times'][:,0]>0
        keys=record['keys'][:,:4];times=(record['times'][:,:1]-1).clamp(min=0).expand_as(keys)
        observed=model.predict_many(keys,fast,torch.ones_like(keys),times).argmax(-1)[mask]
        result['unobserved_past_unknown']['correct']+=int((observed==model.config.unknown).sum())
        result['unobserved_past_unknown']['total']+=observed.numel()
    return {k:{**v,'accuracy':v['correct']/v['total'] if v['total'] else None} for k,v in result.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    args=p.parse_args();directory=Path(args.output_dir)
    if directory.exists():raise FileExistsError('Audit output must be new; retain previous results')
    directory.mkdir(parents=True);torch.set_num_threads(4)
    started=time.perf_counter();model,_,_,data=load_checkpoint(args.checkpoint,args.device)
    result={'checkpoint':str(Path(args.checkpoint).resolve()),
            'sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'metadata':data['metadata'],'seed':1110000,'worlds':64,'main':{},'controls':{}}
    def persist():
        (directory/'audit.partial.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for length in (8,16,32):
        result['main'][str(length)]=evaluate(model,64,length,1110000)
        persist();print(json.dumps({'length':length,'scores':result['main'][str(length)]}),flush=True)
    # Capacity pressure is explicitly outside the training-length range and candidate gate.
    result['pressure64']=evaluate(model,64,64,1110000)
    for control in ('no-write','no-history','shuffled-time')+(
            ('reset-history','reset-current') if model.config.temporal_kind=='split' else ()):
        result['controls'][control]=evaluate(model,64,16,1110000,control)
        persist()
    result['immediate']=immediate(model)
    result['early_diagnostic']=early_sequence(model)
    result['unseen_pairs_diagnostic']=unseen_pairs(model)
    result['restart']=restart_check(args.checkpoint,directory)
    gate_names=('current','history','invalid_pair_unknown','future_unknown')
    result['gate']={'threshold':.95,'passed':all(row[name]['accuracy']>=.95 for row in result['main'].values() for name in gate_names)
        and result['restart']['identical_probabilities'] and result['restart']['continuation_exact_tensors'],
        'failures':[{'length':length,'metric':name,'accuracy':row[name]['accuracy']}
                    for length,row in result['main'].items() for name in gate_names if row[name]['accuracy']<.95]}
    result['seconds']=time.perf_counter()-started;persist()
    (directory/'audit.partial.json').rename(directory/'audit.json')
    print(json.dumps({'completed':str(directory),'gate':result['gate'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
