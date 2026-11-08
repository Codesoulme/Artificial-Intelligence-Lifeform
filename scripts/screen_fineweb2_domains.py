"""Flag suspicious source domains in the mechanically screened FineWeb2 sample."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/p131-fineweb2-screened-sample-20261109/candidate.jsonl"
OUT = ROOT / "runs/p133-fineweb2-domain-screen-20261109"
INPUT_SHA = "27d95c38a4d798aa069a9a97066410ba2285dcde0acbf2c359455d887b698131"
SUSPICIOUS = re.compile(r"porn|sex|xxx|cam|adult|erotic|casino|lottery|cai|bet|博彩|彩票|色情|成人|约炮|裸|av", re.I)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def flags(domain):
    out = []
    if not domain: out.append("missing_domain")
    if len(domain) < 6: out.append("very_short_domain")
    if sum(c.isdigit() for c in domain) / max(1, len(domain)) > .35: out.append("digit_heavy")
    if domain.count(".") >= 4: out.append("deep_subdomain")
    if SUSPICIOUS.search(domain): out.append("suspicious_keyword")
    if re.search(r"[^a-z0-9.-]", domain): out.append("abnormal_host")
    return sorted(set(out))


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    if sha(INPUT) != INPUT_SHA: raise ValueError("input changed")
    rows, counts, domains, flagged_domains = [], Counter(), Counter(), Counter()
    with INPUT.open(encoding="utf-8") as f:
        for line, raw in enumerate(f, 1):
            row = json.loads(raw)
            domain = (urlparse(row.get("url", "")).hostname or "").lower()
            why = flags(domain); domains[domain] += 1
            if why: counts.update(why); flagged_domains[domain] += 1
            rows.append({"source_line_1based": line, "id": row.get("id"), "url": row.get("url"), "domain": domain,
                         "domain_flags": why, "text_sha256": row.get("text_sha256"), "text": row.get("text")})
    OUT.mkdir(parents=False); output = OUT / "domain-review.jsonl"
    with output.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {"status": "domain_screen_pending_review_not_training", "input_sha256": INPUT_SHA,
              "records": len(rows), "flagged_records": sum(1 for r in rows if r["domain_flags"]),
              "flag_counts": dict(counts), "unique_domains": len(domains), "flagged_domains": len(flagged_domains),
              "top_flagged_domains": flagged_domains.most_common(50), "output_sha256": sha(output),
              "training_started": False, "review_approved": False,
              "limitations": ["Domain heuristics are not reputation, legal or semantic judgments.",
                              "A clean domain can host bad content; a flagged domain can host useful content.",
                              "This is one shard sample, not the full corpus."]}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text("# P133 FineWeb2 来源域机械筛选\n\n对 P131 样本做域名级启发式标记，结果只供人工/模型复核，不代表域名信誉或内容安全。当前仍未训练。\n", encoding="utf-8")
    return result


if __name__ == "__main__": print(json.dumps(run(), ensure_ascii=False, indent=2))
