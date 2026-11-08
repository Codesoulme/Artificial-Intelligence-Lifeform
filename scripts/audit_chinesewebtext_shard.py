"""Inventory a pinned ChineseWebText shard without exporting its text."""
import argparse
import collections
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
IDENTITY = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
JUDICIAL = re.compile(r"(?:原告|被告|裁定书|判决书)")
ADVERTISING = re.compile(r"(?:欢迎咨询|招商加盟|微信联系|联系客服|点击购买)")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=521109)
    args = parser.parse_args()
    if args.output.exists() or args.sample <= 0:
        raise ValueError("Output must not exist and sample must be positive")
    expected = "b0713004c9664cbcb40613c51d742b0030bee0fbbdc774f21131f06f3ec222b1"
    source_digest = digest(args.input)
    if source_digest != expected:
        raise ValueError("Unexpected source shard")
    counts = collections.Counter()
    domains = collections.Counter()
    seen = set()
    sample = []
    rng = random.Random(args.seed)
    for batch in pq.ParquetFile(args.input).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            counts["rows"] += 1
            text = row.get("text") or ""
            domain = (row.get("domain") or {}).get("single_label") or "unknown"
            domains[domain] += 1
            counts["characters"] += len(text)
            counts["phone_pattern"] += bool(PHONE.search(text))
            counts["email_pattern"] += bool(EMAIL.search(text))
            counts["identity_pattern"] += bool(IDENTITY.search(text))
            counts["judicial_cue"] += bool(JUDICIAL.search(text))
            counts["advertising_cue"] += bool(ADVERTISING.search(text))
            key = hashlib.sha256(normalize(text).encode("utf8")).digest()
            if key in seen:
                counts["normalized_exact_duplicate_rows"] += 1
            else:
                seen.add(key)
            candidate = {"row_1based": counts["rows"], "domain": domain,
                         "length_characters": len(text), "quality_score": row.get("quality_score"),
                         "text_sha256": hashlib.sha256(text.encode("utf8")).hexdigest()}
            if len(sample) < args.sample:
                sample.append(candidate)
            else:
                index = rng.randrange(counts["rows"])
                if index < args.sample:
                    sample[index] = candidate
    report = {
        "status": "inventory_only_not_approved_for_training",
        "source_sha256": source_digest,
        "source_file": str(args.input),
        "method": "Entire shard; regex cues and normalized exact hash; reservoir sample with fixed seed.",
        "limitations": ["Regex cues are not verified findings or recall guarantees.",
                        "Exact hashes miss near duplicates; no held-out isolation or factual review.",
                        "Sample metadata intentionally excludes raw text."],
        "counts": dict(counts), "domains": dict(domains),
        "sample_seed": args.seed, "sample": sorted(sample, key=lambda row: row["row_1based"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"counts": report["counts"], "domains": report["domains"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
