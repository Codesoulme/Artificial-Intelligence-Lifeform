"""Audit the small authored candidate without training or exporting weights."""
from collections import Counter
import hashlib,json
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"runs/p136-authored-chinese-candidate-20261109/candidate.jsonl"
OUT=ROOT/"runs/p137-authored-candidate-audit-20261109"
TOK=ROOT/"runs/conversation-data-v5/tokenizer.json"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def run():
 if OUT.exists():raise FileExistsError(OUT)
 tok=Tokenizer.from_file(str(TOK)); inp=sha(INPUT); th=sha(TOK); rows=[]; seen=set(); ids=set(); lengths=Counter(); kinds=Counter(); targets=0
 with INPUT.open(encoding="utf-8") as f:
  for line,raw in enumerate(f,1):
   r=json.loads(raw); text=r.get("text",""); rid=r.get("record_id")
   norm="".join(text.split())
   if rid in ids or not text:raise ValueError("duplicate or empty record")
   ids.add(rid); digest=hashlib.sha256(norm.encode()).hexdigest()
   if digest in seen:raise ValueError("normalized duplicate")
   seen.add(digest); n=len(tok.encode(text,add_special_tokens=False).ids); lengths[str(min(1024,n//128*128))]+=1;kinds[r["kind"]]+=1;targets+=max(0,n-1)
   rows.append({"line":line,"record_id":rid,"kind":r["kind"],"text_sha256":hashlib.sha256(text.encode()).hexdigest(),"tokens":n})
 OUT.mkdir(parents=False); (OUT/"records.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for r in rows),encoding="utf-8")
 result={"status":"audited_candidate_not_trained","input_sha256":inp,"tokenizer_sha256":th,"records":len(rows),"unique_normalized_records":len(seen),"kinds":dict(kinds),"token_length_buckets":dict(lengths),"content_tokens":sum(r["tokens"] for r in rows),"next_token_targets":targets,"training_started":False,"training_approval":False,"limitations":["Small authored topic-bound corpus; strong overfit risk.","No independent development isolation or semantic quality review beyond P35 review.","Token counts do not prove chat ability."]}
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return result

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
