"""Build a body-bearing, ignored candidate after domain heuristics; not approved."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"runs/p133-fineweb2-domain-screen-20261109/domain-review.jsonl"
OUT=ROOT/"runs/p134-fineweb2-domain-clean-candidate-20261109"
INPUT_SHA="666988961325114128aebb1321408cce6b8702c8f984f742612ca4205a7b74a8"

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(INPUT)!=INPUT_SHA:raise ValueError("input changed")
 kept=[]; rejected=0
 with INPUT.open(encoding="utf-8") as f:
  for raw in f:
   row=json.loads(raw)
   if row.get("domain_flags"):rejected+=1
   else:kept.append(row)
 OUT.mkdir(parents=False); q=OUT/"candidate.jsonl"
 with q.open("x",encoding="utf-8",newline="\n") as f:
  for row in kept:f.write(json.dumps({k:row.get(k) for k in ("source_line_1based","id","url","domain","text_sha256","text")},ensure_ascii=False,separators=(",",":"))+"\n")
 result={"status":"candidate_not_trained_domain_heuristic_only","input_sha256":INPUT_SHA,"input_records":len(kept)+rejected,"passed":len(kept),"rejected":rejected,"output_sha256":sha(q),"contains_text":True,"training_started":False,"review_approved":False,"limitations":["Domain heuristics are not reputation or semantic review.","This is one FineWeb2 shard sample and cannot represent the corpus.","No cross-corpus isolation, rights or human review completed."]}
 (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (OUT/"report.md").write_text("# P134 FineWeb2 域名启发式候选\n\n从 P131 样本隔离所有触发域名启发式的记录，生成剩余候选。该结果只用于后续审查，不是训练批准。\n",encoding="utf-8")
 return result

if __name__=="__main__":print(json.dumps(run(),ensure_ascii=False,indent=2))
