"""Compare checkpoints on the SAME saved development batches and tokenizer."""
import argparse,json
import numpy as np
import torch
from tokenizers import Tokenizer
from train_conversation_v2 import ROOT,sha,load_rows,monitor
from spora.growing import load
from spora.conversation import restore_conversation
from spora.language_training import frozen_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--plan',default='configs/p23b-conversation.json');p.add_argument('--length',type=int,default=256)
    p.add_argument('--output',required=True);args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));data=ROOT/args.data;vocab=data/'tokenizer.json'
    manifest=json.loads((data/'manifest.json').read_text('utf-8'))
    for name in ('tokenizer.json','pretrain-development.bin','dialogue-development-tokens.jsonl'):
        if sha(data/name)!=manifest['files'][name]:raise ValueError('Development artifact changed')
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    torch.set_num_threads(2);organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
    payload=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=True)
    model=restore_conversation(organism,payload,plan['parent_sha256'],sha(vocab),'cuda')
    tokenizer=Tokenizer.from_file(str(vocab));stream=np.memmap(data/'pretrain-development.bin',dtype=np.uint16,mode='r')
    rows=load_rows(data/'dialogue-development-tokens.jsonl',args.length)
    result=monitor(model,state,tokenizer,stream,rows,args.length,tokenizer.token_to_id('<|pad|>'))
    result.update(checkpoint_sha256=sha(ROOT/args.checkpoint),manifest_sha256=sha(data/'manifest.json'),length=args.length,
        old_individual_exact=before==frozen_digest(organism,state),scope='Same monitor RNG,windows and first32 dialogues at stated length. Development only, no chat score.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)

if __name__=='__main__':main()
