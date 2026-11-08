"""Stream a non-test dialogue-pretrain candidate; never trains or reads test."""
from __future__ import annotations
from collections import Counter
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'runs/clean-dialogue-v2/pretrain-train.jsonl'
MANIFEST=ROOT/'runs/clean-dialogue-v2/manifest.json'
OUT=ROOT/'runs/p158-clean-dialogue-pretrain-audit-20261109'
MANIFEST_SHA='f7c17dec3f6508a44ceab4187583db50ddfa08fd4d687642b997e7e765f79ec1'
PII=re.compile(r'\b(?:1[3-9]\d{9}|\d{17}[\dXx])\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
URL=re.compile(r'https?://|www\.',re.I)
IDENTITY=re.compile(r'我叫|我是由|我的开发者|作为一个模型|我的训练|我的经历|我曾经',re.I)
PROMPT=re.compile(r'^(问题|问：|用户：|助手：|回答：|答：)|\n(?:用户|助手)：')
LOOP=re.compile(r'(.)\1{7,}|(.{3,30})\2{3,}')
HAN=re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('manifest changed')
 counts=Counter();groups=set();sample=[];total_chars=0;total_targets=0
 with INPUT.open(encoding='utf8') as f:
  for line,raw in enumerate(f,1):
   row=json.loads(raw);text=row.get('text','');total=len(text);total_chars+=total;total_targets+=max(0,total-1);counts['records']+=1
   g=row.get('group');
   if g in groups:counts['duplicate_group_rows']+=1
   groups.add(g);counts[f'length_bucket_{min(9,total//500)}']+=1
   compact=''.join(text.split())
   if len(HAN.findall(text))/max(1,len(compact))<.55:counts['low_han_ratio']+=1
   if PII.search(text):counts['pii_pattern']+=1
   if URL.search(text):counts['url']+=1
   if IDENTITY.search(text):counts['identity_or_experience_cue']+=1
   if PROMPT.search(text):counts['prompt_shape']+=1
   if LOOP.search(compact):counts['obvious_repetition']+=1
   key=hashlib.sha256(f'p158\0{row.get("sha256","")}'.encode()).hexdigest()
   if len(sample)<100:sample.append((key,line,row))
   else:
    worst=max(range(len(sample)),key=lambda i:sample[i][0])
    if key<sample[worst][0]:sample[worst]=(key,line,row)
 sample.sort(key=lambda x:(x[0],x[1]));OUT.mkdir(parents=False)
 with (OUT/'sample.jsonl').open('w',encoding='utf8',newline='\n') as f:
  for q,(_,line,row) in enumerate(sample,1):f.write(json.dumps({'queue_line':q,'source_line_1based':line,'text_sha256':row.get('sha256'),'group':row.get('group'),'text':row.get('text','')},ensure_ascii=False,separators=(',',':'))+'\n')
 result={'status':'candidate_not_approved_stream_audit','manifest_sha256':MANIFEST_SHA,'input_sha256':sha(INPUT),'test_split_read':False,'records':counts.pop('records'),'unique_groups':len(groups),'total_characters':total_chars,'next_char_targets':total_targets,'counts':dict(sorted(counts.items())),'sample_records':len(sample),'sample_sha256':sha(OUT/'sample.jsonl'),'training_started':False,'review_approved':False,'limitations':['Rules are not semantic, factual, privacy, rights or independent human review.','The source is a mixed locally licensed noncommercial corpus with upstream quality uncertainty.','Only the train artifact was read; no sealed test content was opened.']}
 (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
