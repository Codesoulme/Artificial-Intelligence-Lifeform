"""Deterministically sample mechanically screened FineWeb2 rows for pending review."""
from collections import Counter
import hashlib, json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/p131-fineweb2-screened-sample-20261109/candidate.jsonl"
OUT = ROOT / "runs/p132-fineweb2-quality-sample-20261109"
INPUT_SHA = "27d95c38a4d798aa069a9a97066410ba2285dcde0acbf2c359455d887b698131"
SEED = "p132-fineweb2-20261109-v1"
SIZE = 100


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def key(row):
    domain = (urlparse(row.get("url", "")).hostname or "").lower()
    length = len(row.get("text", "")); bucket = min(9, length // 500)
    return hashlib.sha256(f"{SEED}\0{domain}\0{bucket}\0{row.get('id','')}".encode()).hexdigest(), domain, str(bucket)


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if sha(INPUT) != INPUT_SHA: raise ValueError("input changed")
    rows = []
    with INPUT.open(encoding="utf-8") as f:
        for line, raw in enumerate(f, 1):
            row = json.loads(raw); rank, domain, bucket = key(row)
            rows.append((rank, line, domain, bucket, row))
    rows.sort(key=lambda x: (x[2], x[3], x[0], x[1]))
    # Round-robin strata, then fill remaining slots by stable global order.
    selected=[]; used=set(); strata=[(d,b) for d,b in sorted({(x[2],x[3]) for x in rows})]
    for d,b in strata:
        for x in rows:
            if len(selected)>=SIZE: break
            if x[2]==d and x[3]==b and x[1] not in used:
                selected.append(x); used.add(x[1]); break
        if len(selected)>=SIZE: break
    for x in sorted(rows, key=lambda y:(y[0],y[1])):
        if len(selected)>=SIZE: break
        if x[1] not in used: selected.append(x); used.add(x[1])
    selected.sort(key=lambda x:x[1])
    OUT.mkdir(parents=False)
    sample=OUT/"samples.jsonl"
    with sample.open("x",encoding="utf-8",newline="\n") as f:
        for q,(rank,line,domain,bucket,row) in enumerate(selected,1):
            f.write(json.dumps({"queue_line":q,"source_line_1based":line,"source_domain":domain,"length_bucket":bucket,"text_sha256":hashlib.sha256(row["text"].encode()).hexdigest(),"text":row["text"]},ensure_ascii=False,separators=(",",":"))+"\n")
    result={"status":"pending_quality_review_not_training","seed":SEED,"records":len(selected),"input_sha256":INPUT_SHA,"output_sha256":sha(sample),"domains":dict(Counter(x[2] for x in selected)),"length_buckets":dict(Counter(x[3] for x in selected)),"contains_text":True,"training_started":False,"limitations":["Sample is drawn from one mechanically screened shard sample, not full FineWeb2.","No rights, semantic, factual or human review completed."]}
    (OUT/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"report.md").write_text("# P132 FineWeb2 质量抽样\n\n机械筛选通过样本中按来源域和长度分层抽取 100 条，正文仅保存在被忽略的 runs 目录。尚未进行独立盲审或训练批准。\n",encoding="utf-8")
    return result


if __name__ == "__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
