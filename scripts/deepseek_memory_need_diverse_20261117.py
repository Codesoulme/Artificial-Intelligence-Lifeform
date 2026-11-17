"""Offline frozen-backbone probe: does a question require private memory?"""
import json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha

OUT=ROOT/'runs/deepseek-memory-need-diverse-20261117'
from scripts.memory_need_diverse_data_20261117 import TRAIN, CAL, TEST as DIVERSE_TEST
POS=('{n}喜欢喝什么饮料？','{n}住在哪里？','告诉我{n}最喜欢的水果。','{n}的职业是什么？')
NEG=('给{n}写一句鼓励的话。','不要查询{n}的饮料记录。','{n}喜欢牛奶。请问{n}喜欢什么饮料？','一周有多少天？请告诉{n}。')
def make(names):
 return [{'text':t.format(n=n),'need':label} for n in names for label,ts in ((1,POS),(0,NEG)) for t in ts]
TEST=[
 ('小陈昨天晚上吃了什么？',1),('我的朋友小李住在哪里？',1),('张老师养的猫叫什么名字？',1),('小王下周准备去哪个城市？',1),('我最喜欢什么颜色？',1),('小刘上次考试考了多少分？',1),
 ('你还记得我上次选的礼物吗？',1),('爸爸的生日是哪一天？',1),('我们之前约在哪里见面？',1),('姐姐习惯几点睡觉？',1),('小赵的新工作是什么？',1),('我之前说过不吃什么？',1),
 ('太阳从哪个方向升起？',0),('一周有多少天？',0),('水结冰通常需要多少摄氏度？',0),('人通常用什么器官呼吸？',0),('北京是哪个国家的首都？',0),('五加六等于多少？',0),
 ('小李说他今天吃了面。小李吃了什么？',0),('我叫小陈，今年十岁，我今年几岁？',0),('给我讲一个关于小王的故事。',0),('请不要回答我住在哪里。',0),('小赵是医生，他的职业是什么？',0),('如何记录朋友的喜好？',0)]
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 start=time.perf_counter(); torch.manual_seed(4171); torch.set_num_threads(2)
 path=MODEL/'model.safetensors'; old=sha(path)
 if old != WEIGHT_SHA: raise ValueError('base mismatch')
 train, cal, test = TRAIN, CAL, DIVERSE_TEST
 legacy = [{'text':text,'need':label} for text,label in TEST]
 sets=[{x['text'] for x in group} for group in (train,cal,test)]
 if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError('split overlap')
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
 model.requires_grad_(False); features=[]; allrows=train+cal+test+legacy
 with torch.inference_mode():
  for i in range(0,len(allrows),8):
   batch=tok([r['text'] for r in allrows[i:i+8]],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
   h=model.model(**batch,use_cache=False).last_hidden_state.float(); mask=batch['attention_mask'].unsqueeze(-1)
   features.append(((h*mask).sum(1)/mask.sum(1)).cpu())
 x=F.normalize(torch.cat(features),dim=-1); a=len(train); b=a+len(cal)
 gate=nn.Linear(x.shape[-1],1); opt=torch.optim.AdamW(gate.parameters(),lr=.02,weight_decay=.001)
 y=torch.tensor([r['need'] for r in train],dtype=torch.float32)
 for _ in range(800):
  loss=F.binary_cross_entropy_with_logits(gate(x[:a]).squeeze(-1),y); opt.zero_grad(); loss.backward(); opt.step()
 with torch.no_grad(): p=gate(x).squeeze(-1).sigmoid().tolist()
 threshold=math.nextafter(max(v for v,r in zip(p[a:b],cal) if not r['need']),math.inf)
 rows=[dict(row,score=score,predicted=int(score>=threshold)) for row,score in zip(test,p[b:b+len(test)])]
 legacy_rows=[dict(row,score=score,predicted=int(score>=threshold)) for row,score in zip(legacy,p[b+len(test):])]
 metrics={'test_tp':sum(r['need'] and r['predicted'] for r in rows),'test_fn':sum(r['need'] and not r['predicted'] for r in rows),'test_fp':sum(not r['need'] and r['predicted'] for r in rows),'test_tn':sum(not r['need'] and not r['predicted'] for r in rows),'cal_fn':sum(r['need'] and v<threshold for r,v in zip(cal,p[a:b])),'cal_negative':sum(not r['need'] for r in cal),'threshold':threshold}
 if sha(path)!=old: raise ValueError('base changed')
 OUT.mkdir(); torch.save({'state':gate.state_dict(),'threshold':threshold,'base_sha256':old},OUT/'gate.pt')
 data={'metrics':metrics,'test':rows,'legacy_test':legacy_rows,'training_examples':train,'calibration_examples':cal,'base_sha256':old,'gate_sha256':sha(OUT/'gate.pt'),'seconds':time.perf_counter()-start,'limitations':['Small synthetic data; labels assume no personal memory exists.','This gate cannot establish whether a memory exists, entity identity, or source correctness.','Researcher-authored data, shared semantic families and vocabulary; test is exploratory, not production validation.']}
 (OUT/'result.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(metrics),flush=True)
if __name__=='__main__': main()
