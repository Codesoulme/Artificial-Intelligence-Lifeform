"""P5 audit uses new streams; diagnostics cannot retune the chosen checkpoint."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spora.checkpoint import load_checkpoint
from spora.p4 import evaluate
from spora.audit_p4 import immediate,early_sequence,unseen_pairs,restart_check
from spora.temporal_tasks import events

@torch.no_grad()
def varied_entities(model,worlds=32,length=64,seed=1311700):
    generator=torch.Generator().manual_seed(seed);device=next(model.parameters()).device
    totals={k:[0,0] for k in ('current','history','invalid','current_unknown')}
    for start in range(0,worlds,4):
        b=min(4,worlds-start)
        keys=torch.stack([torch.randperm(64,generator=generator)[:length] for _ in range(b)]).to(device)
        values=torch.randint(16,(b,length),generator=generator).to(device)
        trust=(torch.rand(b,length,generator=generator)>.25).float().to(device)
        fast,velocity=model.initial_state(b)
        for i in range(length):fast,velocity=model.write_temporal(fast,velocity,keys[:,i],values[:,i],torch.full((b,),i,device=device),trust[:,i])
        mode=torch.ones_like(keys);times=torch.arange(length,device=device)[None].expand_as(keys)
        h=model.predict_many(keys,fast,mode,times).argmax(-1)
        inv=model.predict_many((keys+17)%64,fast,mode,times).argmax(-1)
        cur=model.predict_many(keys,fast).argmax(-1)
        for name,v in {'history':h==values,'invalid':inv==16,'current':cur[trust.bool()]==values[trust.bool()],
                       'current_unknown':cur[~trust.bool()]==16}.items():
            totals[name][0]+=int(v.sum());totals[name][1]+=v.numel()
    return {k:{'correct':a,'total':b,'accuracy':a/b if b else None} for k,(a,b) in totals.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=1310000);args=p.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);torch.set_num_threads(4);start=time.perf_counter()
    model,_,_,data=load_checkpoint(args.checkpoint,'cuda')
    result={'metadata':data['metadata'],'seed':args.seed,'worlds':64,'main':{},'controls':{}}
    for n in (8,16,32,64):
        result['main'][str(n)]=evaluate(model,64,n,args.seed)
        print(json.dumps({'length':n,'metrics':result['main'][str(n)]}),flush=True)
    for c in ('no-write','no-history','shuffled-time','reset-history','reset-current'):
        result['controls'][c]=evaluate(model,32,64,args.seed,c)
    result['immediate']=immediate(model,64,args.seed)
    result['early']=early_sequence(model,64,args.seed+1300)
    result['unseen_pairs']=unseen_pairs(model,64,args.seed+1500)
    result['varied_entities64']=varied_entities(model,32,64,args.seed+1700)
    result['restart']=restart_check(args.checkpoint,output)
    failures=[]
    for n,row in result['main'].items():
        for name in ('current','current_unknown','history','old_history','invalid_pair_unknown','future_unknown'):
            score=row[name]['accuracy']
            if score is not None and score<.95:failures.append({'length':n,'metric':name,'score':score})
    for section in ('early','unseen_pairs'):
        for name,row in result[section].items():
            if row['accuracy'] is not None and row['accuracy']<.95:
                failures.append({'diagnostic':section,'metric':name,'score':row['accuracy']})
    if model.config.current_address=='orthogonal':
        for name,row in result['varied_entities64'].items():
            if row['accuracy'] is not None and row['accuracy']<.95:
                failures.append({'diagnostic':'varied_entities64','metric':name,'score':row['accuracy']})
    result['gate']={'passed':not failures and result['restart']['identical_probabilities'] and result['restart']['continuation_exact_tensors'],
                    'failures':failures}
    result['seconds']=time.perf_counter()-start
    (output/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'complete':str(output),'gate':result['gate']}),flush=True)

if __name__=='__main__':main()
