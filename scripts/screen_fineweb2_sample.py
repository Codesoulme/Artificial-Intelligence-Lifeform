"""Mechanically screen the already-read FineWeb2 sample; never trains."""
from __future__ import annotations
from collections import Counter
import hashlib, json, re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/p128-fineweb2-cmn-multigroup-20261109/sample.jsonl"
INPUT_SHA = "2b0c4e218e7d50b12fb2dd28b9bf8e1358d069606c8bd5d74d909fd90e5a5ba8"
OUT = ROOT / "runs/p131-fineweb2-screened-sample-20261109"

EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+?86[- ]?)?1[3-9]\d{9}")
IDNO = re.compile(r"\b\d{17}[\dXx]\b")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def reasons(row):
    text = row.get("text") or ""
    compact = re.sub(r"\s+", "", text)
    out = []
    if not 100 <= len(text) <= 10000: out.append("length")
    if CONTROL.search(text): out.append("control")
    if EMAIL.search(text): out.append("email_pattern")
    if PHONE.search(text): out.append("phone_pattern")
    if IDNO.search(text): out.append("id_pattern")
    if len(HAN.findall(text)) / max(1, len(compact)) < .55: out.append("low_han_ratio")
    if re.search(r"<[^>]{1,80}>|\[\[|\]\]|```|\{\{", text): out.append("markup")
    if re.search(r"(.)\1{7,}|(.{3,30})\2{3,}", compact): out.append("repetition")
    return sorted(set(out))


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if sha(INPUT) != INPUT_SHA: raise ValueError("input changed")
    rows = []; counts = Counter(); domains = Counter()
    with INPUT.open(encoding="utf-8") as f:
        for line, raw in enumerate(f, 1):
            row = json.loads(raw); why = reasons(row)
            domain = (urlparse(row.get("url", "")).hostname or "").lower()
            domains[domain] += 1
            if why: counts.update(why); counts["rejected_any"] += 1
            else:
                rows.append({**row, "screening": "mechanically_passed", "source_line_1based": line})
    OUT.mkdir(parents=False)
    output = OUT / "candidate.jsonl"
    with output.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {
        "status": "candidate_not_trained_mechanical_screen_only", "input_sha256": INPUT_SHA,
        "input_records": 5000, "passed": len(rows), "rejected": 5000-len(rows),
        "rejection_reasons": dict(sorted(counts.items())), "output_sha256": sha(output),
        "domains_top30": domains.most_common(30), "contains_text": True,
        "review": {"model_blind": False, "human": False, "training_approval": False},
        "limitations": ["Mechanical patterns are not semantic, factual, privacy or rights review.", "Five row groups from one shard are not representative of the corpus.", "Candidate has not been compared with historical or held material."],
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text("# P131 FineWeb2 小样本机械筛选\n\n" + f"输入 5,000 条，机械筛选保留 {len(rows)} 条，剔除 {5000-len(rows)} 条。\n\n" + "这不是质量审查或训练批准；正文仅在被忽略的 runs 目录中。\n", encoding="utf-8")
    return result


if __name__ == "__main__": print(json.dumps(run(), ensure_ascii=False, indent=2))
