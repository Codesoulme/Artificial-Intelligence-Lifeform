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

OUT=ROOT/'runs/deepseek-memory-need-context-pairs-20261117'
PARENT=ROOT/'runs/deepseek-memory-need-diverse-20261117'
PAIRS=[
 ('小芸习惯喝大麦茶。','小芸习惯喝什么？'),
 ('小禾在苏州工作。','小禾在哪座城市工作？'),
 ('我预约了星期二看牙医。','我哪天预约了牙医？'),
 ('弟弟的自行车是绿色的。','弟弟的自行车是什么颜色？'),
 ('阿棠计划十月出发去厦门。','阿棠计划什么时候出发？'),
 ('老师给我们的作业是写观察日记。','老师布置了什么作业？'),
 ('阿遥的猫叫圆圆。','阿遥的猫叫什么名字？'),
 ('妈妈不吃花生。','妈妈不吃什么？'),
 ('上次聚会定在图书馆门口。','上次聚会定在哪里？'),
 ('我选了木制书架。','我选了什么书架？'),
 ('小魏的生日是九月十日。','小魏的生日是几月几日？'),
 ('上周项目会议取消了。','上周项目会议发生了什么？'),
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
 for i,(fact,q) in enumerate(PAIRS):
  rows.extend([{'pair':i,'condition':'question_only','text':q,'need':1},
               {'pair':i,'condition':'context_present','text':fact+'\n'+q,'need':0}])
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
 metrics={'question_only_correct':sum(x['predicted']==1 for x in rows if x['condition']=='question_only'),
          'context_present_correct':sum(x['predicted']==0 for x in rows if x['condition']=='context_present'),
          'both_correct':sum(rows[i]['predicted']==1 and rows[i+1]['predicted']==0 for i in range(0,len(rows),2)),
          'pairs':len(PAIRS),'threshold':threshold}
 if any(sha(p)!=hashes[k] for k,p in paths.items()): raise ValueError('source changed')
 OUT.mkdir(); (OUT/'result.json').write_text(json.dumps({'metrics':metrics,'rows':rows,'source_sha256':hashes,'seconds':time.perf_counter()-start},ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(metrics),flush=True)
if __name__=='__main__': main()
