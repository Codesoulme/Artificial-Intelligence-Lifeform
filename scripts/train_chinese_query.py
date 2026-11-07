"""Bounded P26 development training, without reading final wording results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from spora.chinese_query import ChineseQuery, ChineseQueryConfig, KINDS
from scripts.chinese_query_task import CHARACTERS, examples

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, rows):
    totals = dict(supported=0, parsed_correct=0, correctly_accepted=0, wrong_accepted=0,
                  unsupported=0, correctly_rejected=0)
    failures = []
    for start in range(0,len(rows),128):
        batch=rows[start:start+128]
        parsed=model.parse([r['text'] for r in batch])
        for row,actual in zip(batch,parsed):
            if row['kind']==3:
                totals['unsupported']+=1
                totals['correctly_rejected']+=int(not actual['accepted'])
                ok=not actual['accepted']
            else:
                totals['supported']+=1
                correct=actual['kind']==KINDS[row['kind']] and actual['entity']==row['key'] and actual['event']==row['event']
                totals['parsed_correct']+=int(correct)
                totals['correctly_accepted']+=int(correct and actual['accepted'])
                totals['wrong_accepted']+=int(actual['accepted'] and not correct)
                ok=correct and actual['accepted']
            if not ok and len(failures)<12:
                failures.append(dict(question=row['text'],expected={k:row[k] for k in ('kind','key','event')},actual=actual))
    totals.update(accept_accuracy=totals['correctly_accepted']/max(1,totals['supported']),
        reject_accuracy=totals['correctly_rejected']/max(1,totals['unsupported']),failures=failures)
    totals['passed']=totals['accept_accuracy']>=.98 and totals['reject_accuracy']>=.98
    return totals


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--balanced-times',action='store_true')
    args=p.parse_args()
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    model=ChineseQuery(ChineseQueryConfig(CHARACTERS)).to(args.device)
    rows=examples(args.seed*1000+26,32768,balanced_times=args.balanced_times)
    tokens=model.encode([r['text'] for r in rows])
    intent=torch.tensor([r['kind'] for r in rows],device=args.device)
    digits=torch.tensor([r['digits'] for r in rows],device=args.device)
    positions=torch.tensor([r['positions'] for r in rows],device=args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    small=examples(26001,256,'dev',key_holdout=True,time_holdout=True)
    trace=[];started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        ids=torch.randint(len(rows),(64,),device=args.device)
        batch=tokens[ids]
        length=int((batch!=0).sum(1).max())
        predictions=model(batch[:,:length])
        loss=(F.cross_entropy(predictions['intent'],intent[ids])
            +F.cross_entropy(predictions['digits'].flatten(0,1),digits[ids].flatten())
            +.5*F.cross_entropy(predictions['attention'].flatten(0,1),positions[ids].flatten()))
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        progress=step/args.steps
        for group in opt.param_groups:group['lr']=.0002+.0018*(1+math.cos(math.pi*progress))/2
        opt.step()
        if step%250==0 or step==args.steps:
            model.eval();dev=evaluate(model,small)
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-started,
                development={k:v for k,v in dev.items() if k!='failures'})
            trace.append(row);print(json.dumps(row),flush=True)
    model.eval()
    groups={}
    for index,(keys,times) in enumerate(((False,False),(True,False),(False,True),(True,True))):
        groups[f'held_keys_{keys}_times_{times}']=evaluate(model,examples(26100+index,1024,'dev',keys,times))
    groups['balanced_lengths_held_keys']=evaluate(model,examples(26200,1024,'dev',True,False,True))
    checkpoint=dict(format='spora-chinese-query-v1',config=model.config_dict(),seed=args.seed,
        steps=args.steps,balanced_times=args.balanced_times,state={k:v.detach().cpu() for k,v in model.state_dict().items()})
    torch.save(checkpoint,out/'module.pt')
    report=dict(seed=args.seed,steps=args.steps,balanced_times=args.balanced_times,examples_seen=args.steps*64,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,
        development=groups,trace=trace,passed=all(v['passed'] for v in groups.values()),
        checkpoint_sha256=fingerprint(out/'module.pt'),
        source_sha256={name:fingerprint(ROOT/name) for name in ('spora/chinese_query.py','scripts/chinese_query_task.py','scripts/train_chinese_query.py')},
        scope='Finite neural Chinese address extraction, digits and closed template combinations. No memory answers are in the training targets. No final wording test opened.')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],accuracies={k:v['accept_accuracy'] for k,v in groups.items()},seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
