"""Check actual trained GPU incremental inference against complete prefixes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from spora.growing import load
from spora.conversation import restore_conversation
from spora.language_training import frozen_digest
from scripts.prepare_conversation_v2 import ROOT,sha
from scripts.converse_research import prefix_tokens

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/'configs/p23c-conversation.json').read_text('utf-8'));vocab=ROOT/plan['data']/'tokenizer.json'
    torch.set_num_threads(2);core,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(core,state)
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    payload=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=True)
    model=restore_conversation(core,payload,plan['parent_sha256'],sha(vocab),'cuda');tok=Tokenizer.from_file(str(vocab))
    cases=json.loads((ROOT/'data/conversation-development.json').read_text('utf-8'))['scenarios']
    rows=[]
    for case in cases:
        prefix=prefix_tokens(tok,[('user',case['turns'][0])]);tokens=torch.tensor([prefix],device='cuda');past=None;max_difference=0.;agreement=True
        for i in range(24):
            complete=model(tokens,state)[0,-1]
            incremental,past=model(tokens if i==0 else tokens[:,-1:],state,past=past,use_past=True);incremental=incremental[0,-1]
            max_difference=max(max_difference,float((complete-incremental).abs().max()))
            agreement=agreement and int(complete.argmax())==int(incremental.argmax())
            tokens=torch.cat((tokens,complete.argmax().reshape(1,1)),1)
        rows.append(dict(scenario=case['name'],argmax_equal=agreement,maximum_logit_difference=max_difference))
    result=dict(checkpoint_sha256=sha(ROOT/args.checkpoint),rows=rows,all_argmax_equal=all(r['argmax_equal'] for r in rows),
        old_individual_exact=before==frozen_digest(core,state),scope='24 generated positions per12 development prompts; actual trained GPU path and original nonempty memory. Floating point differences reported, no answer repair.')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    if not result['all_argmax_equal']:raise SystemExit(1)

if __name__=='__main__':main()
