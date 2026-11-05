"""Select on development only; evaluate locked policies on a fresh audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from .checkpoint import load_checkpoint,save_checkpoint
from .p2 import evaluate,stream,write
from .evaluate import isolated_query


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--phase',choices=['development','audit'],required=True)
    parser.add_argument('--worlds',type=int,default=32)
    parser.add_argument('--selection')
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    destination=output/(args.phase+'.json')
    if destination.exists():raise FileExistsError('Preserve completed evaluation')
    torch.set_num_threads(4)
    model,_,_,parent=load_checkpoint(args.checkpoint,'cpu')
    started=time.perf_counter();rows=[]
    if args.phase=='development':
        candidates=[('one',.8),('two',.8)]+[('adaptive-max4',x) for x in (.5,.8,.95)]
        for policy,threshold in candidates:
            scores=[]
            for pairs in (8,64):
                scores.append(evaluate(model,worlds=args.worlds,pairs=pairs,seed=510000+pairs,
                                       policy=policy,threshold=threshold))
            # Pre-audit scalarization penalizes damage and adds a small update-cost penalty.
            score=sum((x['corrected']+x['unrelated']+x['initial'])/3
                      -.01*x['inner_updates_per_trusted_event'] for x in scores)/len(scores)
            row={'policy':policy,'threshold':threshold,'scores':scores,'selection_score':score}
            rows.append(row);print(json.dumps(row),flush=True)
        winner=max(rows,key=lambda x:x['selection_score'])
        selected={'policy':winner['policy'],'threshold':winner['threshold']}
    else:
        if not args.selection:raise ValueError('Audit requires frozen development selection')
        development=json.loads(Path(args.selection).read_text(encoding='utf-8'))
        selected=development['selected']
        expected=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        if expected!=development['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
        # Include fixed baselines even if adaptive was selected; no audit-driven reselection.
        policies=[('one',.8),('two',.8)]
        if (selected['policy'],selected['threshold']) not in policies:
            policies.append((selected['policy'],selected['threshold']))
        for policy,threshold in policies:
            scores=[]
            for pairs in (8,16,32,64):
                scores.append(evaluate(model,worlds=args.worlds,pairs=pairs,seed=610000+pairs,
                                       policy=policy,threshold=threshold))
            row={'policy':policy,'threshold':threshold,'scores':scores}
            rows.append(row);print(json.dumps(row),flush=True)
    result={'phase':args.phase,'checkpoint':str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            'training':parent['metadata'],'worlds':args.worlds,'selected':selected,
            'results':rows,'seconds':time.perf_counter()-started}
    if args.phase=='audit':
        generator=torch.Generator().manual_seed(619000)
        events,keys,_,current=stream(generator,1,16,model.config,'cpu')
        fast,velocity=model.initial_state(1)
        for event in events:
            fast,velocity,_=write(model,fast,velocity,event,selected['policy'],selected['threshold'])
        with torch.no_grad():expected=model.predict_many(keys,fast).argmax(-1).flatten().tolist()
        individual=output/'individual.pt'
        save_checkpoint(individual,model,fast,velocity,cursor=len(events),
                        metadata={'stage':'P2 audit individual','parent_sha256':result['checkpoint_sha256'],**selected})
        restored=isolated_query(individual,keys.flatten().tolist())['predictions']
        reset=isolated_query(individual,keys.flatten().tolist(),True)['predictions']
        if expected!=restored:raise AssertionError('Restart changed predictions')
        result['restart']={'match':True,'expected':expected,'restored':restored,
                           'reset_fast':reset,'truth':current.flatten().tolist()}
        result['caveats']=['Environmental trust gate is not learned source judgment',
                           'Current-value semantics; no historical version query implemented',
                           'Per-seed exploratory comparisons; do not count events as independent training seeds']
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'saved':str(destination),'selected':selected,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
