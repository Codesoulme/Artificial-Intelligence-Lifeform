"""Sample a screened FineWeb2 candidate for pending review."""
import hashlib,json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"runs/p134-fineweb2-domain-clean-candidate-20261109/candidate.jsonl"
OUT=ROOT/"runs/p135-fineweb2-quality-sample-20261109"
INPUT_SHA="5c3616d28739e95cc2443699fd1af5db90eab2908adbbed5a04fd480c57a75f6"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(INPUT)!=INPUT_SHA:raise ValueError("input changed")
 rows=[]
 with INPUT.open(encoding="utf-8") as f:
  for line,raw in enumerate(f,1):
   r=json.loads(raw); d=(urlparse(r.get("url","")).hostname or "").lower(); b=min(9,len(r.get("text", ""))//500)
   k=hashlib.sha256(f"p135\0{d}\0{b}\0{r.get('id','')}".encode()).hexdigest();rows.append((d,b,k,line,r))
 rows.sort(key=lambda x:(x[0],x[1],x[2],x[3]));selected=[];used=set()
 for d,b,k,line,r in rows:
  if len(selected)>=100:break
  if (d,b) not in used: selected.append((d,b,k,line,r));used.add((d,b))
 for d,b,k,line,r in sorted(rows,key=lambda x:(x[2],x[3])):
  if len(selected)>=100:break
  if line not in {x[3] for x in selected}:selected.append((d,b,k,line,r))
 selected.sort(key=lambda x:x[3]);OUT.mkdir(parents=False);q=OUT/"samples.jsonl"
 with q.open("x",encoding="utf-8",newline="\n") as f:
  for i,(d,b,k,line,r) in enumerate(selected,1):f.write(json.dumps({"queue_line":i,"source_line_1based":line,"domain":d,"length_bucket":b,"text_sha256":r.get("text_sha256"),"text":r.get("text")},ensure_ascii=False,separators=(",",":"))+"\n")
 result={"status":"pending_quality_review_not_training","input_sha256":INPUT_SHA,"records":len(selected),"output_sha256":sha(q),"domains":dict(Counter(x[0] for x in selected)),"length_buckets":dict(Counter(x[1] for x in selected)),"training_started":False,"review_approved":False}
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return result

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
