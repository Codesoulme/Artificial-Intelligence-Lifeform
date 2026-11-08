"""Build a fixed, flagged review queue from the P158 sample; never trains."""
from __future__ import annotations
import hashlib,json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'runs/p158-clean-dialogue-pretrain-audit-20261109/sample.jsonl'
OUT=ROOT/'runs/p159-clean-dialogue-review-queue-20261109'
INPUT_SHA='11ee64690b855e24db12e6b43c456c6a35479765d8eced9cc8ed7f995a89495f'
HAN=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]')
PII=re.compile(r'\b(?:1[3-9]\d{9}|\d{17}[\dXx])\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
URL=re.compile(r'https?://|www\.',re.I)
IDENTITY=re.compile(r'我叫|我是由|我的开发者|作为一个模型|我的训练|我的经历|我曾经',re.I)
PROMPT=re.compile(r'^(问题|问：|用户：|助手：|回答：|答：)|\n(?:用户|助手)：')
LOOP=re.compile(r'(.)\1{7,}|(.{3,30})\2{3,}')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def flags(text):
 compact=''.join(text.split());out=[]
 if len(HAN.findall(text))/max(1,len(compact))<.55:out.append('low_han_ratio')
 if PII.search(text):out.append('pii_pattern')
 if URL.search(text):out.append('url')
 if IDENTITY.search(text):out.append('identity_or_experience_cue')
 if PROMPT.search(text):out.append('prompt_shape')
 if LOOP.search(compact):out.append('obvious_repetition')
 return out
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(INPUT)!=INPUT_SHA:raise ValueError('sample changed')
 rows=[];counts=Counter()
 for raw in INPUT.read_text(encoding='utf8').splitlines():
  row=json.loads(raw); why=flags(row['text']); strata='flagged' if why else 'mechanically_clean'; counts[strata]+=1
  rows.append({**row,'review_status':'pending_independent_review','review_stratum':strata,'mechanical_flags':why,'length_bucket':min(9,len(row['text'])//500)})
 OUT.mkdir(parents=False); q=OUT/'queue.jsonl'
 q.write_text('\n'.join(json.dumps(x,ensure_ascii=False,separators=(',',':')) for x in rows)+'\n',encoding='utf8')
 result={'status':'pending_independent_review_not_training','input_sha256':INPUT_SHA,'records':len(rows),'strata':dict(counts),'length_buckets':dict(Counter(str(x['length_bucket']) for x in rows)),'queue_sha256':sha(q),'training_started':False,'review_approved':False,'limitations':['Mechanical flags are not content or rights judgments.','The queue is a 100-row sample from one training artifact, not a corpus estimate.','No final test content was read.']}
 (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
