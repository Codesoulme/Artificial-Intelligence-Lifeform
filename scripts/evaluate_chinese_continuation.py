"""Raw continuation probes separate language modeling from dialogue protocol."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from spora.conversation import restore_conversation, generate
from spora.conversation_sampling import generate_sampled
from spora.conversation_metrics import repetition_metrics
from spora.growing import load
from spora.language_training import frozen_digest
from scripts.prepare_conversation_v2 import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True);p.add_argument('--budget',type=int,default=64)
    args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'))
    checkpoint=ROOT/args.checkpoint;parent=ROOT/plan['parent'];vocab=ROOT/plan['data']/'tokenizer.json'
    before_file=sha(checkpoint)
    if sha(parent)!=plan['parent_sha256']:raise ValueError('Parent changed')
    torch.set_num_threads(2)
    core,state,_=load(parent,'cuda');before=frozen_digest(core,state)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=restore_conversation(core,payload,plan['parent_sha256'],sha(vocab),'cuda')
    tokenizer=Tokenizer.from_file(str(vocab));eos=tokenizer.token_to_id('<|eos|>')
    bos=tokenizer.token_to_id('<|bos|>')
    forbidden=tuple(tokenizer.token_to_id(s) for s in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>'))
    cases_path=ROOT/'data/chinese-continuation-development.json'
    cases=json.loads(cases_path.read_text('utf-8'))
    rows=[]
    for i,case in enumerate(cases['cases']):
        prefix=[bos]+tokenizer.encode(case['prefix']).ids
        for policy in ('greedy','sampled'):
            torch.cuda.synchronize();start=time.perf_counter()
            if policy=='greedy':result=generate(model,state,prefix,eos,args.budget,forbidden)
            else:result=generate_sampled(model,state,prefix,eos,args.budget,forbidden,seed=271108+i)
            torch.cuda.synchronize()
            text=tokenizer.decode(result['tokens'],skip_special_tokens=False)
            rows.append(dict(**case,policy=policy,text=text,seconds=time.perf_counter()-start,
                **result,**repetition_metrics(text),utf8_valid='\ufffd' not in text,useful=None))
    report=dict(scope=cases['scope'],checkpoint_sha256=before_file,cases_sha256=sha(cases_path),
        rows=rows,old_individual_exact=before==frozen_digest(core,state),checkpoint_unchanged=before_file==sha(checkpoint),
        evaluator_sha256=sha(Path(__file__)),source_sha256={name:sha(ROOT/name) for name in ('spora/conversation.py','spora/conversation_state.py','spora/conversation_sampling.py')})
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(cases=len(rows),old_individual_exact=report['old_individual_exact'],checkpoint_unchanged=report['checkpoint_unchanged'])))


if __name__=='__main__':main()
