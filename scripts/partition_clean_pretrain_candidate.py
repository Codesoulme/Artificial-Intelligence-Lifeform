"""Partition a large text candidate by use shape; metadata and samples only."""
from __future__ import annotations
from collections import Counter
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'runs/clean-dialogue-v2/pretrain-train.jsonl'
OUT=ROOT/'runs/p161-clean-pretrain-partition-20261109'
INPUT_SHA='4ec0b001db65b7ccabfaa064e56ce6dff3bbf0bb0a3890c116f4b99999bae9a1'
HAN=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]')
CODE=re.compile(r'```|\b(?:def|class|import|function|var|const|SELECT)\b|<\/?[A-Za-z][^>]*>',re.I)
TASK=re.compile(r'^(?:请|帮我|帮助|如何|怎样|给出|写一|生成|总结|翻译|判断|解释|根据|列出|完成|设计|描述|输出|输入：|问题：|问：)',re.I)
TASK_ANY=re.compile(r'(?:输入(?:内容|文本)?|问题：|回答：|答案：|正确答案|总结如下|请将|请对|请根据|以下(?:文本|句子|数字|内容))',re.I)
ROLE=re.compile(r'(?:^|\n)(?:用户|助手|system|assistant|user)\s*[:：]|(?:^|\n)[A-Z][A-Za-z]{0,12}\s*[:：]')
COMMERCIAL=re.compile(r'产品|广告|购买|客服|优惠|促销|招商|加盟|品牌|销量|官网|报价|订单|营销|推广',re.I)
HIGH_STAKES=re.compile(r'医疗|疾病|药物|诊断|治疗|法律|合同|诉讼|投资|股票|贷款|金融|疫情|新冠|战争|枪击|选举',re.I)
PII=re.compile(r'\b(?:1[3-9]\d{9}|\d{17}[\dXx])\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
LOOP=re.compile(r'(.)\1{7,}|(.{3,30})\2{3,}')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def classify(text):
 compact=''.join(text.split()); han=len(HAN.findall(text))/max(1,len(compact)); flags=[]
 if CODE.search(text):flags.append('code_or_markup')
 if TASK.search(text) or TASK_ANY.search(text[:240]):flags.append('instruction_shape')
 if ROLE.search(text):flags.append('dialogue_or_role_shape')
 if COMMERCIAL.search(text):flags.append('commercial_terms')
 if HIGH_STAKES.search(text):flags.append('high_stakes_terms')
 if PII.search(text):flags.append('pii_pattern')
 if LOOP.search(compact):flags.append('repetition')
 if han<.55:flags.append('low_han_ratio')
 if len(text)<100:flags.append('short')
 if flags and flags[0] in ('code_or_markup','instruction_shape','dialogue_or_role_shape','commercial_terms'):
  category=flags[0]
 elif 'pii_pattern' in flags or 'repetition' in flags or 'low_han_ratio' in flags:
  category='risk_or_noise'
 elif 'high_stakes_terms' in flags:
  category='knowledge_risk'
 elif not flags and len(text)>=100 and han>=.75:
  category='prose_candidate'
 else: category='mixed_review'
 return category,flags
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(INPUT)!=INPUT_SHA:raise ValueError('input changed')
 counts=Counter();samples={};total=0
 with INPUT.open(encoding='utf8') as f:
  for line,raw in enumerate(f,1):
   row=json.loads(raw);category,flags=classify(row.get('text',''));total+=1;counts[category]+=1
   bucket=samples.setdefault(category,[])
   key=hashlib.sha256(f'p161\0{row.get("sha256","")}'.encode()).hexdigest()
   item=(key,line,row,flags)
   if len(bucket)<20:bucket.append(item)
   else:
    worst=max(range(len(bucket)),key=lambda i:bucket[i][0])
    if key<bucket[worst][0]:bucket[worst]=item
 OUT.mkdir(parents=False);sample=OUT/'samples.jsonl'
 with sample.open('w',encoding='utf8',newline='\n') as out:
  for category,bucket in sorted(samples.items()):
   for q,(key,line,row,flags) in enumerate(sorted(bucket),1):
    out.write(json.dumps({'category':category,'sample_index':q,'source_line_1based':line,'text_sha256':row.get('sha256'),'group':row.get('group'),'flags':flags,'text':row.get('text','')},ensure_ascii=False,separators=(',',':'))+'\n')
 result={'status':'partitioned_candidate_not_training','input_sha256':INPUT_SHA,'records':total,'categories':dict(sorted(counts.items())),'sample_sha256':sha(sample),'samples_per_category':20,'test_split_read':False,'training_started':False,'training_approval':False,'limitations':['Pattern partition is not semantic, factual, privacy or rights review.','Prose candidate means only mechanical shape, not approval.','Only train artifact was read; no development or sealed test text was opened.']}
 (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
