"""Finite Unicode-character training with strict lineage and frozen organism."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from .growing import load
from .codepoint import CodepointVocabulary,CodepointLanguageBridge,sha,windows
from .language_bridge import causal_loss,PAD
from .language_training import ByteTrainer,load_interface,atomic_save,frozen_digest
from .corpus_admission import validate_language_training_approval

def main():
 p=argparse.ArgumentParser();p.add_argument('--individual',required=True);s=p.add_mutually_exclusive_group(required=True);s.add_argument('--interface');s.add_argument('--resume')
 p.add_argument('--manifest',required=True);p.add_argument('--vocab',required=True);p.add_argument('--review',required=True);p.add_argument('--output',required=True);p.add_argument('--start-language-training',action='store_true');p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=4);p.add_argument('--length',type=int,default=128);p.add_argument('--seed',type=int,default=2710000);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--save-every',type=int,default=10);p.add_argument('--context-layers',type=int,default=2);a=p.parse_args()
 if not a.start_language_training:p.error('Explicit --start-language-training required')
 if not 1<=a.steps<=10000 or not 1<=a.batch<=16 or not 2<=a.length<=256 or not 0<=a.context_layers<=4: p.error('Invalid finite budget or context')
 out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 docs=json.loads(Path(a.manifest).read_text(encoding='utf8'))
 from .text_data import read_manifest
 documents=read_manifest(a.manifest);vocab=CodepointVocabulary.from_file(a.vocab)
 parent=Path(a.individual);parent_sha=sha(parent);manifest_sha=sha(Path(a.manifest));vocab_sha=sha(Path(a.vocab))
 review=json.loads(Path(a.review).read_text(encoding='utf8'))
 approval=validate_language_training_approval(review,manifest_sha256=manifest_sha,parent_sha256=parent_sha,schedule_steps=a.steps)
 train,dev=[windows(documents,vocab,split,a.length) for split in ('train','development')]
 organism,state,_=load(parent,a.device);before=frozen_digest(organism,state);model=CodepointLanguageBridge(organism,vocab,a.context_layers).to(a.device)
 provenance={'manifest_sha256':manifest_sha,'vocab_sha256':vocab_sha,'review_sha256':sha(Path(a.review)),'context_layers':a.context_layers,'documents':[{k:v for k,v in d.items() if k!='text'} for d in documents],'seed':a.seed,'batch':a.batch,'length':a.length,'device':a.device}
 trainer=ByteTrainer(model,state,parent_sha,provenance,a.seed,format_name='spora-codepoint-training-v1')
 source=Path(a.resume or a.interface);payload=torch.load(source,map_location='cpu',weights_only=True)
 if a.resume:trainer.restore(payload)
 else:
  if payload.get('format')!='spora-codepoint-interface-v1' or payload.get('trained_tokens')!=0 or payload.get('vocab_sha256')!=vocab_sha:raise ValueError('Use a matching pristine codepoint interface')
  load_interface(model,payload,parent_sha)
 if trainer.steps+a.steps>10000:raise ValueError('Total pilot budget exceeds 10000 steps')
 out.mkdir(parents=True);meta={'arguments':vars(a),'parent_sha256':parent_sha,'provenance':provenance,'approval':approval,'source_checkpoint_sha256':sha(source),'limitations':'Unicode character prediction; organism frozen; no chat claim'}
 (out/'config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');started=time.perf_counter()
 with (out/'metrics.jsonl').open('w',encoding='utf8') as log:
  for i in range(a.steps):
   loss=trainer.step(train,a.batch);final=i==a.steps-1
   if trainer.steps%a.save_every==0 or final:atomic_save(trainer.snapshot(),out/'checkpoint.pt')
   if trainer.steps%10==0 or final:
    total=count=0.
    with torch.no_grad():
     for row in dev:
      batch=row[None].to(a.device);weight=int((batch[:,1:]!=PAD).sum());total+=float(causal_loss(model(batch,state),batch))*weight;count+=weight
    record={'step':trainer.steps,'train_loss':loss,'development_loss':total/count,'tokens_seen':trainer.tokens_seen,'seconds_this_run':time.perf_counter()-started};log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
 if sha(parent)!=parent_sha or frozen_digest(organism,state)!=before:raise AssertionError('Protected parent changed')
 (out/'complete.json').write_text(json.dumps({'steps':trainer.steps,'tokens_seen':trainer.tokens_seen,'parent_unchanged':True,'all_organism_tensors_unchanged':True,'organism_tensor_digest':before})+'\n',encoding='utf8')
if __name__=='__main__':main()
