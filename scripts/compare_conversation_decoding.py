"""Paired development-only decoder comparison, same weights and prompts."""
import argparse,json,sys,re,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from spora.growing import load
from spora.conversation import restore_conversation,generate
from spora.conversation_sampling import generate_sampled
from spora.language_training import frozen_digest
from scripts.prepare_conversation_v2 import ROOT,sha
from scripts.train_conversation_v2 import PROMPTS
from scripts.converse_research import prefix_tokens

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',default='configs/p23c-conversation.json');args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));vocab=ROOT/plan['data']/'tokenizer.json'
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    torch.set_num_threads(2);organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
    payload=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=True)
    model=restore_conversation(organism,payload,plan['parent_sha256'],sha(vocab),'cuda')
    tokenizer=Tokenizer.from_file(str(vocab));eos=tokenizer.token_to_id('<|eos|>')
    forbidden=tuple(tokenizer.token_to_id(x) for x in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>'))
    rows=[]
    for number,prompt in enumerate(PROMPTS):
        prefix=prefix_tokens(tokenizer,[('user',prompt)])
        for policy in ('greedy','sampled'):
            started=time.perf_counter()
            if policy=='greedy':result=generate(model,state,prefix,eos,96,forbidden)
            else:result=generate_sampled(model,state,prefix,eos,96,forbidden,temperature=.7,top_p=.9,seed=231123+number)
            text=tokenizer.decode(result['tokens'],skip_special_tokens=False)
            rows.append(dict(policy=policy,prompt=prompt,text=text,**result,seconds=time.perf_counter()-started,utf8_valid='\ufffd' not in text,obvious_loop=bool(re.search(r'(.{2,20})\1{3,}',text))))
    result=dict(checkpoint_sha256=sha(ROOT/args.checkpoint),step=payload['steps'],temperature=.7,top_p=.9,seed_base=231123,
        old_individual_exact=before==frozen_digest(organism,state),rows=rows,
        scope='Development diagnostic; same weights and prompts. No repetition penalty, replacement responses, or generated-text repair. Changing decode does not establish language competence.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({p:dict(ended=sum(r['ended_with_eos'] for r in rows if r['policy']==p),loops=sum(r['obvious_loop'] for r in rows if r['policy']==p)) for p in ('greedy','sampled')}),flush=True)

if __name__=='__main__':main()
