"""Build a bounded, unapproved Chinese web-text candidate for content review.

This is a preprocessing experiment, not a guarantee of privacy or correctness.
It reads a single fixed source shard and never loads final chat acceptance items.
"""
import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "b0713004c9664cbcb40613c51d742b0030bee0fbbdc774f21131f06f3ec222b1"
DOMAINS = {"education", "encyclopedia", "general"}
PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
ID_NUMBER = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
CUES = re.compile(r"原告|被告|裁定书|判决书|身份证|家庭住址|联系电话|手机号|欢迎咨询|招商加盟|微信联系|联系客服|点击购买|招聘启事|中奖名单")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def reject_reason(text, domain, prompts):
    if domain not in DOMAINS:
        return "domain"
    if not 240 <= len(text) <= 6000:
        return "length"
    normalized = unicodedata.normalize("NFKC", text)
    if PHONE.search(normalized) or EMAIL.search(normalized) or ID_NUMBER.search(normalized) or CUES.search(normalized):
        return "sensitive_or_promotional_cue"
    if "\ufffd" in text or re.search(r"(.{2,20})\1{3,}", text):
        return "corruption_or_repetition"
    if sum("\u4e00" <= ch <= "\u9fff" for ch in normalized) / len(normalized) < 0.55:
        return "low_chinese_ratio"
    compact = norm(text)
    if any(p in compact for p in prompts):
        return "development_prompt_exact"
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(args.source) != SOURCE_SHA:
        raise ValueError("Unexpected source shard")
    cases_path = ROOT / "data/conversation-development.json"
    cases = json.loads(cases_path.read_text(encoding="utf8"))
    prompts = [norm(q) for case in cases["scenarios"] for q in case["turns"] if len(norm(q)) >= 12]
    args.output.mkdir(parents=True)
    counts = collections.Counter()
    seen = set()
    with (args.output / "candidate.jsonl").open("x", encoding="utf8") as out:
        for batch in pq.ParquetFile(args.source).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                text = row.get("text") or ""
                domain = (row.get("domain") or {}).get("single_label") or "unknown"
                reason = reject_reason(text, domain, prompts)
                if reason:
                    counts["rejected_" + reason] += 1
                    continue
                normalized = norm(text)
                key = hashlib.sha256(normalized.encode("utf8")).hexdigest()
                if key in seen:
                    counts["rejected_exact_duplicate"] += 1
                    continue
                seen.add(key)
                out.write(json.dumps({"source_row_1based": counts["source_rows"], "domain": domain,
                                      "sha256_normalized": key, "text": text}, ensure_ascii=False) + "\n")
                counts["candidate_rows"] += 1
                counts["candidate_" + domain] += 1
    manifest = {
        "status": "candidate_not_approved_for_training",
        "source_sha256": SOURCE_SHA,
        "source_revision": "8007a3ed1f6b9351f82afb2643f8bb6ff094b029",
        "development_cases_sha256": sha(cases_path),
        "candidate_sha256": sha(args.output / "candidate.jsonl"),
        "script_sha256": sha(Path(__file__)),
        "counts": dict(counts),
        "limitations": ["Heuristic filtering cannot establish quality, accuracy, rights or privacy.",
                        "Only normalized exact duplicates and full development prompts are filtered.",
                        "No near-duplicate, cross-split, or held-out corpus overlap audit has been completed.",
                        "Content requires independent sampling and review before training."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
