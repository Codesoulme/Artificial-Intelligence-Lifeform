"""Conservative body-quality screening for a FineWeb2 candidate; never trains."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/p134-fineweb2-domain-clean-candidate-20261109/candidate.jsonl"
OUT = ROOT / "runs/p149-fineweb2-body-quality-candidate-20261109"
INPUT_SHA = "5c3616d28739e95cc2443699fd1af5db90eab2908adbbed5a04fd480c57a75f6"

URL = re.compile(r"https?://|www\\.", re.I)
REPEATED = re.compile(r"(.)\\1{7,}|(.{3,30})\\2{3,}")
SUSPICIOUS = re.compile(r"博彩|彩票|赌博|色情|成人|约炮|裸聊|兼职刷单|私彩|六合彩", re.I)
PROMOTION = re.compile(r"点击|注册|登录|客服|优惠|折扣|招商|加盟|代理|官网|免费|热线|二维码", re.I)
SITE_CHROME = re.compile(r"版权与免责声明|免责声明|本站原创|责任编辑|转载.*注明|联系我们", re.I)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quality_flags(text: str) -> list[str]:
    """Return only high-signal body flags; absence is not a quality claim."""
    text = text or ""
    flags = []
    if len(URL.findall(text)) >= 2:
        flags.append("url_burst")
    if len(SUSPICIOUS.findall(text)) >= 2:
        flags.append("suspicious_terms")
    if len(PROMOTION.findall(text)) >= 4:
        flags.append("commercial_template")
    if SITE_CHROME.search(text):
        flags.append("site_chrome")
    compact = re.sub(r"\\s+", "", text)
    if REPEATED.search(compact):
        flags.append("repetition")
    return sorted(set(flags))


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(INPUT) != INPUT_SHA:
        raise ValueError("input changed")
    kept = []
    counts = Counter()
    records = 0
    with INPUT.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            flags = quality_flags(row.get("text", ""))
            records += 1
            if flags:
                counts.update(flags)
            else:
                kept.append({**row, "body_screening": "mechanically_passed"})
    OUT.mkdir(parents=False)
    output = OUT / "candidate.jsonl"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {
        "status": "candidate_not_trained_body_heuristic_only",
        "input_sha256": INPUT_SHA,
        "input_records": records,
        "passed": len(kept),
        "rejected": records - len(kept),
        "flag_counts": dict(sorted(counts.items())),
        "output_sha256": sha(output),
        "training_started": False,
        "review_approved": False,
        "limitations": [
            "Body heuristics are not semantic, factual, privacy, rights or human review.",
            "This is one shard sample and cannot represent FineWeb2.",
            "A mechanically clean row may still be unsuitable or legally unresolved.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
