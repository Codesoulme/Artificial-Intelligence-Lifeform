"""Finite future language training. Explicit flag required; not run by readiness."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from .growing import load
from .language_bridge import LanguageBridge,causal_loss,PAD
from .text_data import read_manifest,windows

def main():
    p=argparse.ArgumentParser();p.add_argument('--individual',required=True);p.add_argument('--interface',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start-language-training',action='store_true');p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--length',type=int,default=128)
    p.add_argument('--seed',type=int,default=1910000);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if not a.start_language_training:p.error('Explicit --start-language-training required; readiness does not train text')
    if not 1<=a.steps<=10000 or not 1<=a.batch<=16 or not 2<=a.length<=256:p.error('Invalid finite budget')
    output=Path(a.output)
    if output.exists():raise FileExistsError('Choose a new trial directory')
    documents=read_manifest(a.manifest);train=windows(documents,'train',a.length);dev=windows(documents,'development',a.length)
    # Test text is validated for provenance/deduplication, never used for updates or model selection here.
    digest=hashlib.sha256(Path(a.individual).read_bytes()).hexdigest()
    payload=torch.load(a.interface,map_location='cpu',weights_only=True)
    if payload['organism_sha256']!=digest:raise ValueError('Interface belongs to a different individual')
    torch.set_num_threads(2);torch.manual_seed(a.seed);generator=torch.Generator().manual_seed(a.seed)
    organism,state,_=load(a.individual,a.device);model=LanguageBridge(organism).to(a.device)
    merged=model.state_dict();merged.update(payload['state']);model.load_state_dict(merged)
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=3e-4,weight_decay=.01)
    output.mkdir(parents=True);started=time.perf_counter();tokens_seen=0
    metadata={'arguments':vars(a),'parent_sha256':digest,'source_interface_sha256':hashlib.sha256(Path(a.interface).read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
              'documents':[{k:v for k,v in d.items() if k!='text'} for d in documents],
              'limitations':'Local byte prediction trial; no semantic event writer, frozen organism'}
    (output/'config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'metrics.jsonl').open('w') as log:
        for step in range(a.steps):
            ids=torch.randint(len(train),(a.batch,),generator=generator);batch=train[ids].to(a.device)
            optimizer.zero_grad(set_to_none=True);loss=causal_loss(model(batch,state),batch)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
            tokens_seen+=int((batch[:,1:]!=PAD).sum())
            if (step+1)%10==0 or step==a.steps-1:
                with torch.no_grad():validation=sum(float(causal_loss(model(d[None].to(a.device),state),d[None].to(a.device))) for d in dev)/len(dev)
                row={'step':step+1,'train_loss':float(loss.detach()),'development_loss':validation,'tokens_seen':tokens_seen,'seconds':time.perf_counter()-started}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('organism.')}
    temporary=output/'interface.pt.tmp';torch.save({'format':'spora-byte-interface-v1','organism_sha256':digest,'state':state_dict,
        'trained_tokens':tokens_seen,'optimizer':optimizer.state_dict()},temporary);os.replace(temporary,output/'interface.pt')
    if hashlib.sha256(Path(a.individual).read_bytes()).hexdigest()!=digest:raise AssertionError('Parent file changed')
    (output/'complete.json').write_text(json.dumps({'steps':a.steps,'tokens_seen':tokens_seen,'parent_unchanged':True}))

if __name__=='__main__':main()
