"""Broader fixed development likelihood; no final-test access or chat score."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tokenizers import Tokenizer
from spora.growing import load
from spora.conversation import restore_conversation
from spora.conversation_data import DialogueRows
from spora.language_training import frozen_digest
from scripts.train_conversation_v2 import padded,text_batch,batch_loss
from scripts.prepare_conversation_v2 import ROOT,sha

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',default='runs/conversation-data-v4')
    p.add_argument('--output',required=True);p.add_argument('--plan',default='configs/p23c-conversation.json');args=p.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    data=ROOT/args.data;manifest=json.loads((data/'manifest.json').read_text('utf-8'));plan=json.loads((ROOT/args.plan).read_text('utf-8'))
    for name in ('tokenizer.json','pretrain-development.bin','dialogue-development-tokens.jsonl'):
        if sha(data/name)!=manifest['files'][name]:raise ValueError('Development input changed')
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    torch.set_num_threads(2);core,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(core,state)
    payload=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=True)
    model=restore_conversation(core,payload,plan['parent_sha256'],manifest['tokenizer_sha256'],'cuda')
    tok=Tokenizer.from_file(str(data/'tokenizer.json'));pad=tok.token_to_id('<|pad|>')
    dataset=DialogueRows(data/'dialogue-development-tokens.jsonl',512);records=[];total=0.;targets=0
    buckets={'short':dict(loss_sum=0.,targets=0,rows=0),'long':dict(loss_sum=0.,targets=0,rows=0)}
    for i in range(len(dataset)):
        row=dataset[i];length=len(row['tokens'])-1;x,y=padded([row],length,pad,'cuda');n=int((y!=-100).sum());loss=float(batch_loss(model,state,x,y))
        records.append(dict(group=row['group'],nll=loss,targets=n,tokens=length+1));total+=loss*n;targets+=n
        bucket=buckets['short' if length<=256 else 'long'];bucket['loss_sum']+=loss*n;bucket['targets']+=n;bucket['rows']+=1
    for value in buckets.values():value['nll']=value.pop('loss_sum')/max(1,value['targets'])
    stream=np.memmap(data/'pretrain-development.bin',mode='r',dtype=np.uint16);rng=random.Random(231301);text_total=0.;text_count=0
    for i in range(64):
        x,y=text_batch(stream,rng,4,512,'cuda');n=y.numel();text_total+=float(batch_loss(model,state,x,y))*n;text_count+=n
    result=dict(checkpoint_sha256=sha(ROOT/args.checkpoint),source_step=payload['steps'],manifest_sha256=sha(data/'manifest.json'),
        dialogue=dict(nll=total/targets,rows=len(dataset),targets=targets,buckets=buckets,records=records),
        pretrain=dict(nll=text_total/text_count,targets=text_count,seed=231301,windows=256),
        old_individual_exact=before==frozen_digest(core,state),
        scope='All encodable DEVELOPMENT dialogues and256 deterministic512-token development text windows. No test generation; teacher-forced likelihood does not measure actual chat usefulness.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(dialogue_nll=total/targets,rows=len(dataset),pretrain_nll=text_total/text_count,buckets=buckets)),flush=True)

if __name__=='__main__':main()
