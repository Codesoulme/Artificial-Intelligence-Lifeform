"""Audit a fixed memory-need classifier on matched contextual evidence pairs."""
import json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha

OUT=ROOT/'runs/deepseek-memory-need-hard-context-20261117'
PARENT=ROOT/'runs/deepseek-memory-need-paired-train-20261117'
CASES=[
 ('阿琪喜欢茉莉花。','阿琪住在海边。','阿波喜欢茉莉花。','阿琪不喜欢茉莉花。','阿琪喜欢什么花？'),
 ('小石周末去成都。','小石养了一只鸟。','小武周末去成都。','小石周末不去成都。','小石周末去哪座城市？'),
 ('姑姑习惯六点起床。','姑姑爱看画展。','舅舅习惯六点起床。','姑姑不习惯六点起床。','姑姑习惯几点起床？'),
 ('阿晴的书包是紫色的。','阿晴买了一把雨伞。','阿乔的书包是紫色的。','阿晴的书包不是紫色的。','阿晴的书包是什么颜色？'),
 ('老周的职业是电工。','老周喜欢踢球。','老梁的职业是电工。','老周的职业不是电工。','老周的职业是什么？'),
 ('我的行李放在门后。','我买了三本杂志。','他的行李放在门后。','我的行李没有放在门后。','我的行李放在哪里？'),
 ('阿雪选择了素面。','阿雪正在学游泳。','阿霖选择了素面。','阿雪没有选择素面。','阿雪选择了什么食物？'),
 ('妹妹的演出在周日。','妹妹喜欢画漫画。','哥哥的演出在周日。','妹妹的演出不在周日。','妹妹的演出在星期几？'),
]

def main():
 if OUT.exists(): raise FileExistsError(OUT)
 start=time.perf_counter(); torch.set_num_threads(2)
 paths={'base':MODEL/'model.safetensors','candidate':PARENT/'gate.pt','report':PARENT/'result.json'}
 hashes={k:sha(p) for k,p in paths.items()}
 report=json.loads(paths['report'].read_text(encoding='utf8'))
 if hashes['base']!=WEIGHT_SHA or hashes['candidate']!=report['gate_sha256']: raise ValueError('source mismatch')
 checkpoint=torch.load(paths['candidate'],map_location='cpu',weights_only=True)
 if checkpoint['base_sha256']!=WEIGHT_SHA: raise ValueError('base mismatch')
 rows=[]
 for i,(fact,other,wrong,neg,q) in enumerate(CASES):
  for label,context,need in [('relevant',fact,0),('same_subject_other',other,1),('wrong_subject',wrong,1),('negated',neg,1)]:
   rows.append({'case':i,'condition':label,'text':context+'\n'+q,'need':need})
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
 model.requires_grad_(False); feats=[]
 with torch.inference_mode():
  for i in range(0,len(rows),8):
   batch=tok([r['text'] for r in rows[i:i+8]],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
   h=model.model(**batch,use_cache=False).last_hidden_state.float(); mask=batch['attention_mask'].unsqueeze(-1)
   feats.append(((h*mask).sum(1)/mask.sum(1)).cpu())
 x=F.normalize(torch.cat(feats),dim=-1)
 gate=nn.Linear(x.shape[-1],1); gate.load_state_dict(checkpoint['state'],strict=True)
 with torch.no_grad(): scores=gate(x).squeeze(-1).sigmoid().tolist()
 threshold=checkpoint['threshold']
 for row,score in zip(rows,scores): row.update(score=score,predicted=int(score>=threshold))
 metrics={kind:sum(r['predicted']==r['need'] for r in rows if r['condition']==kind) for kind in ('relevant','same_subject_other','wrong_subject','negated')}
 metrics.update(all_four=sum(all(rows[i+j]['predicted']==rows[i+j]['need'] for j in range(4)) for i in range(0,len(rows),4)),cases=len(CASES),threshold=threshold)
 if any(sha(p)!=hashes[k] for k,p in paths.items()): raise ValueError('source changed')
 OUT.mkdir(); (OUT/'result.json').write_text(json.dumps({'metrics':metrics,'rows':rows,'source_sha256':hashes,'seconds':time.perf_counter()-start},ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(metrics),flush=True)
if __name__=='__main__': main()
