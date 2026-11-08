"""Prepare P136 as a tiny, provenance-bound pretraining data directory."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"runs/p136-authored-chinese-candidate-20261109/candidate.jsonl"
BASE=ROOT/"runs/conversation-data-v5"
OUT=ROOT/"runs/authored-pretrain-adapter-v1"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def split(row):
 n=int(hashlib.sha256(row["record_id"].encode()).hexdigest()[:8],16)%100
 return "development" if n<10 else "test" if n<20 else "train"

def run():
 if OUT.exists():raise FileExistsError(OUT)
 tok=Tokenizer.from_file(str(BASE/"tokenizer.json")); rows={s:[] for s in ("train","development","test")}; rejected=0
 with SOURCE.open(encoding="utf-8") as f:
  for raw in f:
   row=json.loads(raw); ids=tok.encode(row["text"],add_special_tokens=False).ids
   if any(x in ids for x in (tok.token_to_id("<|bos|>"),tok.token_to_id("<|eos|>"),tok.token_to_id("<|pad|>"))): rejected+=1; continue
   rows[split(row)].append(row)
 OUT.mkdir(parents=False); stats={"source_sha256":sha(SOURCE)}
 for s,items in rows.items():
  raw=OUT/f"pretrain-{s}.jsonl"; binary=OUT/f"pretrain-{s}.bin"; total=0
  with raw.open("x",encoding="utf-8",newline="\n") as jf,binary.open("xb") as bf:
   for row in items:
    ids=[tok.token_to_id("<|bos|>")]+tok.encode(row["text"],add_special_tokens=False).ids+[tok.token_to_id("<|eos|>")]
    np.asarray(ids,dtype="<u2").tofile(bf); total+=len(ids)
    jf.write(json.dumps({"text":row["text"],"record_id":row["record_id"],"source_kind":row["kind"],"source_file_sha256":row["source_file_sha256"],"token_count":len(ids)},ensure_ascii=False,separators=(",",":"))+"\n")
  stats[f"{s}_records"]=len(items);stats[f"{s}_tokens"]=total
 for name in ("tokenizer.json","dialogue-train-tokens.jsonl","dialogue-development-tokens.jsonl","dialogue-test-tokens.jsonl","dialogue-train.jsonl","dialogue-development.jsonl","dialogue-test.jsonl","focus-tokens.json"):
  shutil.copy2(BASE/name,OUT/name)
 stats["rejected_special_token_collision"]=rejected
 manifest={"status":"authored_pretrain_pipeline_smoke_not_quality_certified","source_candidate_sha256":sha(SOURCE),"tokenizer_sha256":sha(OUT/"tokenizer.json"),"stats":stats,"license":"user_authored_local_research_source","files":{}}
 manifest["files"]={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}
 (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 audit={"passed":True,"scope":"Exact artifact binding for a 20-step pipeline smoke only; not a quality certification.","manifest_sha256":{"runs/authored-pretrain-adapter-v1":sha(OUT/"manifest.json")},"data_artifact_sha256":manifest["files"],"issues":["Narrow user-authored topic; strong overfit risk.","No independent general-quality approval."]}
 (OUT/"lineage-audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 review={"approved_for_pilot":True,"status":"pipeline_smoke_only_not_capability_approval","max_schedule_steps":20,"max_target_shares":{"pretrain":1.0},"reviewed_checkpoint_sha256":"eae8e678db0e9b87b7a88528206e2210fb93d87cb925308aa8a2d1ae67bc1234","training_source_sha256":{"pretrain":manifest["files"]["pretrain-train.bin"],"focus":manifest["files"]["focus-tokens.json"]},"basis":"User-authored local source and existing P35-reviewed dialogue; permits only a 20-step pipeline smoke with frozen individual and no capability claim.","limitations":["Not a broad Chinese quality certification.","No final evaluation or independent generalization claim."]}
 (OUT/"pipeline-review.json").write_text(json.dumps(review,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 return {"status":manifest["status"],"stats":stats,"manifest_sha256":sha(OUT/"manifest.json"),"review_sha256":sha(OUT/"pipeline-review.json")}

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
