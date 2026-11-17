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
OUT=ROOT/'runs/deepseek-memory-need-hard-train-20261117'
from scripts.deepseek_memory_need_hard_context_20261117 import CASES as HARD_TEST
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
HARD_TRAIN=[
 ('小乔喜欢桂花。','小乔喜欢阅读。','小米喜欢桂花。','小乔不喜欢桂花。','小乔喜欢什么花？'),
 ('哥哥计划去西安。','哥哥买了个茶杯。','姐姐计划去西安。','哥哥不打算去西安。','哥哥计划去哪个城市？'),
 ('阿榆的书桌是白色的。','阿榆种了一盆花。','阿枫的书桌是白色的。','阿榆的书桌不是白色的。','阿榆的书桌是什么颜色？'),
 ('张老师下午三点开会。','张老师喜欢摄影。','王老师下午三点开会。','张老师不在下午三点开会。','张老师几点开会？'),
 ('阿朗从事园艺。','阿朗有一个书架。','阿宁从事园艺。','阿朗不从事园艺。','阿朗从事什么工作？'),
 ('我把茶叶放在抽屉里。','我喜欢听钢琴曲。','她把茶叶放在抽屉里。','我没有把茶叶放在抽屉里。','我的茶叶放在哪里？'),
 ('小海喜欢喝果汁。','小海喜欢唱歌。','小星喜欢喝果汁。','小海不喜欢喝果汁。','小海喜欢喝什么？'),
 ('阿辰在周三出发。','阿辰常常骑车。','阿文在周三出发。','阿辰不在周三出发。','阿辰哪天出发？'),
]
HARD_CAL=[
 ('小松种的是番茄。','小松有一只白猫。','小安种的是番茄。','小松种的不是番茄。','小松种的是什么？'),
 ('爸爸的座位在前排。','爸爸会弹吉他。','妈妈的座位在前排。','爸爸的座位不在前排。','爸爸的座位在哪里？'),
 ('阿福预约周六出游。','阿福喜欢书法。','阿平预约周六出游。','阿福没有预约周六出游。','阿福预约哪天出游？'),
 ('小芸的宠物是仓鼠。','小芸喜欢跳舞。','小燕的宠物是仓鼠。','小芸的宠物不是仓鼠。','小芸养了什么宠物？'),
]
def hard_make(cases):
 rows=[]
 for i,(fact,other,wrong,neg,q) in enumerate(cases):
  for label,context,need in [('relevant',fact,0),('same_subject_other',other,1),('wrong_subject',wrong,1),('negated',neg,1)]:
   rows.append({'pair':i,'condition':label,'text':context+'\n'+q,'need':need})
 return rows

def main():
 if OUT.exists(): raise FileExistsError(OUT)
 t=time.perf_counter();torch.manual_seed(4171);torch.set_num_threads(2)
 path=MODEL/'model.safetensors'; basehash=sha(path)
 if basehash!=WEIGHT_SHA: raise ValueError('base mismatch')
 train=make(TRAIN)+hard_make(HARD_TRAIN)+OTHER_TRAIN;cal=make(CAL)+hard_make(HARD_CAL)+OTHER_CAL;test=hard_make(HARD_TEST)
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
 stats={name:sum(r['predicted']==r['need'] for r in rows if r['condition']==name) for name in ('relevant','same_subject_other','wrong_subject','negated')}
 stats.update(all_four=sum(all(rows[i+j]['predicted']==rows[i+j]['need'] for j in range(4)) for i in range(0,len(rows),4)),threshold=threshold,cal_positive_missed=sum(p<threshold for p,r in zip(probs[a:b],cal) if r['need']))
 if sha(path)!=basehash: raise ValueError('base changed')
 OUT.mkdir();torch.save({'state':gate.state_dict(),'threshold':threshold,'base_sha256':basehash},OUT/'gate.pt')
 (OUT/'result.json').write_text(json.dumps({'metrics':stats,'test':rows,'train':train,'calibration':cal,'base_sha256':basehash,'gate_sha256':sha(OUT/'gate.pt'),'seconds':time.perf_counter()-t},ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
