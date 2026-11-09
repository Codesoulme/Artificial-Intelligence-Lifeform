"""Save unassisted multi-turn development responses, never auto-certify chat."""
import argparse, collections, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from spora.growing import load
from spora.conversation import restore_conversation, generate
from spora.conversation_sampling import generate_sampled
from spora.conversation_metrics import repetition_metrics
from spora.language_training import frozen_digest
from scripts.prepare_conversation_v2 import sha
from scripts.converse_research import prefix_tokens
ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate_cases(model, state, tokenizer, cases, budget=96, policy='greedy', seed=231123):
    eos=tokenizer.token_to_id('<|eos|>')
    forbidden=tuple(tokenizer.token_to_id(x) for x in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>'))
    rows=[]
    for case in cases['scenarios']:
        history=[]
        for turn,prompt in enumerate(case['turns'],1):
            history.append(('user',prompt));removed=0
            while len(prefix_tokens(tokenizer,history))+budget>model.config.context and len(history)>1:
                history=history[2:];removed+=2
            prefix=prefix_tokens(tokenizer,history);device=next(model.parameters()).device
            if device.type=='cuda':torch.cuda.synchronize(device)
            start=time.perf_counter()
            if policy=='greedy':value=generate(model,state,prefix,eos,budget,forbidden)
            elif policy=='sampled':value=generate_sampled(model,state,prefix,eos,budget,forbidden,temperature=.7,top_p=.9,seed=seed+len(rows))
            else:raise ValueError('Unknown generation policy')
            if device.type=='cuda':torch.cuda.synchronize(device)
            seconds=time.perf_counter()-start;text=tokenizer.decode(value['tokens'],skip_special_tokens=False)
            row=dict(scenario=case['name'],turn=turn,prompt=prompt,text=text,**value,seconds=seconds,
                input_tokens=len(prefix),removed_history_messages=removed,utf8_valid='\ufffd' not in text,
                **repetition_metrics(text),useful=None)
            rows.append(row);history.append(('assistant',text))
        print(json.dumps(dict(scenario=case['name'],turns=len(case['turns']))),flush=True)
    if not rows:raise ValueError('No development turns')
    summary=dict(turns=len(rows),eos_fraction=sum(r['ended_with_eos'] for r in rows)/len(rows),
        valid_utf8_fraction=sum(r['utf8_valid'] for r in rows)/len(rows),
        obvious_loops=sum(r['obvious_loop'] for r in rows),history_trimmed_turns=sum(bool(r['removed_history_messages']) for r in rows),
        mean_seconds=sum(r['seconds'] for r in rows)/len(rows),
        useful_fraction=None,status='ungraded_development_not_acceptance')
    return summary,rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',default='configs/p23b-conversation.json')
    p.add_argument('--cases',default='data/conversation-development.json')
    p.add_argument('--budget',type=int,default=96)
    p.add_argument('--policy',choices=('greedy','sampled'),default='greedy')
    p.add_argument('--seed',type=int,default=231123)
    args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));path=ROOT/args.checkpoint
    parent=ROOT/plan['parent'];vocab=ROOT/plan['data']/'tokenizer.json'
    if sha(parent)!=plan['parent_sha256']:raise ValueError('Individual changed')
    digest=sha(path);torch.set_num_threads(2)
    organism,state,_=load(parent,'cuda');before=frozen_digest(organism,state)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model=restore_conversation(organism,payload,sha(parent),sha(vocab),'cuda')
    tokenizer=Tokenizer.from_file(str(vocab))
    cases=json.loads((ROOT/args.cases).read_text('utf-8'))
    summary,rows=evaluate_cases(model,state,tokenizer,cases,args.budget,args.policy,args.seed)
    result=dict(checkpoint_sha256=digest,step=payload['steps'],plan_sha256=sha(ROOT/args.plan),
        policy=args.policy,sampling=dict(temperature=.7,top_p=.9,seed_base=args.seed) if args.policy=='sampled' else None,
        tokenizer_sha256=sha(vocab),cases_sha256=sha(ROOT/args.cases),evaluator_sha256=sha(Path(__file__)),
        generation_source_sha256={name:sha(ROOT/name) for name in ('spora/conversation.py','spora/conversation_sampling.py','scripts/converse_research.py','spora/conversation_metrics.py')},
        repetition_metric_version='direct_and_numbered_lines_v2',
        old_individual_exact=before==frozen_digest(organism,state),checkpoint_unchanged=sha(path)==digest,
        summary=summary,rows=rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
