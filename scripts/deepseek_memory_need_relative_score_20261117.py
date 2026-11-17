"""Development-only relative score audit, including no-correct-candidate sets."""
import json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_memory_need_hard_train_20261117 import HARD_CAL
from scripts.deepseek_memory_need_hard_context_20261117 import CASES
OUT=ROOT/'runs/deepseek-memory-need-relative-score-20261117'
SOURCES={'paired':ROOT/'runs/deepseek-memory-need-paired-train-20261117', 'hard':ROOT/'runs/deepseek-memory-need-hard-train-20261117'}
def prepare(cases):
 rows=[]
 for i,(right,other,wrong,neg,q) in enumerate(cases):
  rows.append({'case':i,'condition':'none','text':q})
  for kind,fact in [('right',right),('other',other),('wrong',wrong),('negated',neg)]: rows.append({'case':i,'condition':kind,'text':fact+'\n'+q})
 return rows
def decisions(rows,scores,cut):
 groups=[]
 for i in range(0,len(rows),5):
  r=rows[i:i+5];p=scores[i:i+5];delta=[p[0]-x for x in p[1:]]
  winner=max(range(4),key=lambda j:delta[j]); bad=max(range(1,4),key=lambda j:delta[j])
  groups.append({'case':r[0]['case'],'deltas':dict(zip(('right','other','wrong','negated'),delta)),
                 'winner':('right','other','wrong','negated')[winner],
                 'accepted_with_correct':delta[winner]>cut,
                 'accepted_without_correct':delta[bad]>cut})
 return groups
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 start=time.perf_counter();torch.set_num_threads(2)
 paths={'base':MODEL/'model.safetensors'}
 for name,p in SOURCES.items(): paths[name+'_gate']=p/'gate.pt';paths[name+'_report']=p/'result.json'
 hashes={k:sha(v) for k,v in paths.items()}
 if hashes['base']!=WEIGHT_SHA: raise ValueError('base mismatch')
 gates={}
 for name in SOURCES:
  report=json.loads(paths[name+'_report'].read_text(encoding='utf8'))
  if hashes[name+'_gate']!=report['gate_sha256']:raise ValueError('gate mismatch')
  payload=torch.load(paths[name+'_gate'],map_location='cpu',weights_only=True)
  if payload['base_sha256']!=WEIGHT_SHA:raise ValueError('gate base mismatch')
  gates[name]=payload['state']
 cal,test=prepare(HARD_CAL),prepare(CASES);rows=cal+test
 if {r['text'] for r in cal}&{r['text'] for r in test}: raise ValueError('exact overlap')
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
 feats=[]
 with torch.inference_mode():
  for i in range(0,len(rows),8):
   batch=tok([r['text'] for r in rows[i:i+8]],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
   h=model.model(**batch,use_cache=False).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
   feats.append(((h*mask).sum(1)/mask.sum(1)).cpu())
 x=F.normalize(torch.cat(feats),dim=-1);result={}
 for name,state in gates.items():
  gate=nn.Linear(x.shape[-1],1);gate.load_state_dict(state,strict=True)
  with torch.no_grad(): logits=gate(x).squeeze(-1).tolist()
  c=decisions(cal,logits[:len(cal)],0)
  cut=max(g['deltas'][k] for g in c for k in ('other','wrong','negated'))
  test_groups=decisions(test,logits[len(cal):],cut)
  result[name]={'calibration_cut':cut,'calibration_cases':c,'test_cases':test_groups,'metrics':{
   'correct_winner':sum(g['winner']=='right' for g in test_groups),
   'correct_winner_accepted':sum(g['winner']=='right' and g['accepted_with_correct'] for g in test_groups),
   'wrong_winner_accepted':sum(g['winner']!='right' and g['accepted_with_correct'] for g in test_groups),
   'rejected_with_correct':sum(not g['accepted_with_correct'] for g in test_groups),
   'accepted_without_correct':sum(g['accepted_without_correct'] for g in test_groups)}}
 if any(sha(p)!=hashes[k] for k,p in paths.items()):raise ValueError('source changed')
 OUT.mkdir();(OUT/'result.json').write_text(json.dumps({'result':result,'source_sha256':hashes,'seconds':time.perf_counter()-start},ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({k:v['metrics'] for k,v in result.items()}),flush=True)
if __name__=='__main__':main()
