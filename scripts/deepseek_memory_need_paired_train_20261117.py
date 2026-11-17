"""Paired context training probe; no persistent memory or deployed router."""
import json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from scripts.deepseek_local_baseline import MODEL,WEIGHT_SHA,sha
from scripts.deepseek_memory_need_context_pairs_20261117 import PAIRS as TEST_PAIRS
from scripts.memory_need_diverse_data_20261117 import TRAIN as OTHER_TRAIN,CAL as OTHER_CAL
OUT=ROOT/'runs/deepseek-memory-need-paired-train-20261117'
TRAIN=[
 ('阿文养了一只兔子。','阿文养了什么动物？'),('小雨明天要去长沙。','小雨明天要去哪？'),
 ('我把钥匙放在鞋柜上。','我的钥匙放哪了？'),('父亲最爱听民乐。','父亲最爱听什么音乐？'),
 ('小春是摄影师。','小春的职业是什么？'),('奶奶喜欢吃南瓜。','奶奶喜欢吃什么？'),
 ('我决定周五出发。','我决定哪天出发？'),('小彤昨天去参观博物馆。','小彤昨天去了哪里？'),
 ('哥哥买了一把蓝色的雨伞。','哥哥买的雨伞是什么颜色？'),('陈叔叔的狗叫豆豆。','陈叔叔的狗叫什么？'),
 ('朋友给我送了一本画册。','朋友送了我什么？'),('昨天我们的会议提前结束。','昨天会议怎么样了？'),
]
CAL=[('小菲的外套是黄色的。','小菲的外套是什么颜色？'),('姐姐上周去了上海。','姐姐上周去了哪？'),('阿彬喜欢爬山。','阿彬喜欢什么活动？'),('我的车停在东门。','我的车停在哪里？')]
def make(pairs):
 rows=[]
 for i,(fact,q) in enumerate(pairs):
  other=pairs[(i+1)%len(pairs)][0]
  rows.extend([{'pair':i,'condition':'question_only','text':q,'need':1},
               {'pair':i,'condition':'relevant','text':fact+'\n'+q,'need':0},
               {'pair':i,'condition':'unrelated','text':other+'\n'+q,'need':1}])
 return rows
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 t=time.perf_counter();torch.manual_seed(4171);torch.set_num_threads(2)
 path=MODEL/'model.safetensors'; basehash=sha(path)
 if basehash!=WEIGHT_SHA: raise ValueError('base mismatch')
 train=make(TRAIN)+OTHER_TRAIN;cal=make(CAL)+OTHER_CAL;test=make(TEST_PAIRS)
 split=[{r['text'] for r in x} for x in (train,cal,test)]
 if any(split[i]&split[j] for i in range(3) for j in range(i+1,3)): raise ValueError('split overlap')
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
 feats=[];allrows=train+cal+test
 with torch.inference_mode():
  for i in range(0,len(allrows),8):
   batch=tok([r['text'] for r in allrows[i:i+8]],padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
   h=model.model(**batch,use_cache=False).last_hidden_state.float();mask=batch['attention_mask'].unsqueeze(-1)
   feats.append(((h*mask).sum(1)/mask.sum(1)).cpu())
 x=F.normalize(torch.cat(feats),dim=-1);a=len(train);b=a+len(cal)
 gate=nn.Linear(x.shape[-1],1);opt=torch.optim.AdamW(gate.parameters(),lr=.02,weight_decay=.001)
 y=torch.tensor([r['need'] for r in train],dtype=torch.float32)
 for _ in range(800):
  loss=F.binary_cross_entropy_with_logits(gate(x[:a]).squeeze(-1),y);opt.zero_grad();loss.backward();opt.step()
 with torch.no_grad(): probs=gate(x).squeeze(-1).sigmoid().tolist()
 threshold=math.nextafter(max(p for p,r in zip(probs[a:b],cal) if not r['need']),math.inf)
 rows=[dict(r,score=p,predicted=int(p>=threshold)) for r,p in zip(test,probs[b:])]
 stats={name:sum(r['predicted']==r['need'] for r in rows if r['condition']==name) for name in ('question_only','relevant','unrelated')}
 stats.update(pair_both=sum(rows[i]['predicted']==1 and rows[i+1]['predicted']==0 for i in range(0,len(rows),3)),all_three=sum(all(rows[i+j]['predicted']==rows[i+j]['need'] for j in range(3)) for i in range(0,len(rows),3)),threshold=threshold,cal_positive_missed=sum(p<threshold for p,r in zip(probs[a:b],cal) if r['need']))
 if sha(path)!=basehash: raise ValueError('base changed')
 OUT.mkdir();torch.save({'state':gate.state_dict(),'threshold':threshold,'base_sha256':basehash},OUT/'gate.pt')
 (OUT/'result.json').write_text(json.dumps({'metrics':stats,'test':rows,'train':train,'calibration':cal,'base_sha256':basehash,'gate_sha256':sha(OUT/'gate.pt'),'seconds':time.perf_counter()-t},ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
