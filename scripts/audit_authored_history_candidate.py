"""Audit the expanded authored candidate."""
import hashlib,json
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1];INPUT=ROOT/"runs/p145-authored-chinese-history-candidate-20261109/candidate.jsonl";TOK=ROOT/"runs/conversation-data-v5/tokenizer.json";OUT=ROOT/"runs/p146-authored-history-audit-20261109"
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def run():
 if OUT.exists():raise FileExistsError(OUT)
 tok=Tokenizer.from_file(str(TOK));seen=set();kinds=Counter();lengths=Counter();tokens=0;records=0
 with INPUT.open(encoding="utf-8") as f:
  for raw in f:
   r=json.loads(raw);text=r["text"];norm="".join(text.split());d=hashlib.sha256(norm.encode()).hexdigest()
   if d in seen:raise ValueError("duplicate")
   seen.add(d);ids=tok.encode(text,add_special_tokens=False).ids;tokens+=len(ids);lengths[str(min(1024,len(ids)//128*128))]+=1;kinds[r["kind"]]+=1;records+=1
 OUT.mkdir(parents=False);result={"status":"audited_authored_history_candidate_not_trained","input_sha256":sha(INPUT),"records":records,"tokens":tokens,"next_token_targets":tokens-records,"kinds":dict(kinds),"length_buckets":dict(lengths),"unique_normalized":len(seen),"training_approval":False,"training_started":False}
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");return result
if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
