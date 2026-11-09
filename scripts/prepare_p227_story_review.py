"""Prepare a local second blind-review queue.

Run: python scripts/prepare_p227_story_review.py [--output runs/NEW_DIRECTORY]
Only the pinned P120 sample and four P121 review files are read. No evaluation
data or model is loaded, and this does not authorize or start training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "runs/p120-story-quality-sample-20261109/samples.jsonl"
SAMPLE_SHA = "b29dd2db7ba4f0601d9e524300207daa4eaaaa720b0f0e66f28b79481e401213"
SHARDS = tuple(ROOT / f"runs/p120-story-quality-review-agent{i}.json" for i in range(1, 5))
SHARD_SHAS = (
    "3c8b63eccf2b7854208bbc2d4e5eccb414a1a9edb6b261f7798fc78bec5d8a14",
    "059c46ac1e8a31db0ff0e280d0265908bdd00a0d502cf1fa6953cca80978141e",
    "63d48a83f773063de83212afa6d2f255f0e37b532c92bc21529451db4fa22582",
    "b160b1df97c574aee94975e3234d1f660c60aef55ffba984a8601dc809c8f853",
)
OUTPUT = ROOT / "runs/p227-story-review-20261109"
SEED = 12020261109
SAMPLE_COUNT = 200
SHARD_COUNT = 50
CLEAN_COUNT = 81
LABEL_COUNTS = {"clean": 81, "minor": 87, "major": 32}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pinned(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f"digest mismatch: {path}")
    return data


def validated_sample(data: bytes) -> dict:
    rows = {}
    unique = {key: set() for key in ("source_row_1based", "input_line_1based", "normalized_sha256", "group_sha256", "source_original_sha256")}
    for number, line in enumerate(data.splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"sample line {number}: invalid text")
        text = row["text"]
        normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()
        full, group, original = (sha(normalized.encode()), sha(normalized[:96].encode()), sha(text.encode()))
        if (row.get("normalized_sha256"), row.get("group_sha256"), row.get("source_original_sha256")) != (full, group, original):
            raise ValueError(f"sample line {number}: text identity mismatch")
        for field, seen in unique.items():
            value = row.get(field)
            if field.endswith("_1based") and (type(value) is not int or value < 1):
                raise ValueError(f"sample line {number}: invalid {field}")
            if value in seen:
                raise ValueError(f"sample line {number}: duplicate {field}")
            seen.add(value)
        tokens = row.get("content_tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError(f"sample line {number}: invalid token count")
        stratum = ("000-127" if tokens < 128 else "128-255" if tokens < 256 else
                   "256-383" if tokens < 384 else "384-511" if tokens < 512 else "512-plus")
        if row.get("stratum") != stratum:
            raise ValueError(f"sample line {number}: stratum mismatch")
        identity = sha(f"{SEED}\0{stratum}\0{row['source_row_1based']}\0{group}\0{full}".encode())
        if row.get("sample_id") != identity or row.get("rank_sha256") != identity or identity in rows:
            raise ValueError(f"sample line {number}: invalid or duplicate sample identity")
        rows[identity] = row
    if len(rows) != SAMPLE_COUNT:
        raise ValueError("sample count mismatch")
    return rows


def run(output: Path = OUTPUT) -> dict:
    output = Path(output).resolve()
    runs = (ROOT / "runs").resolve(strict=True)
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("output must be a new directory below runs")
    if output.exists():
        raise FileExistsError(output)
    if len(SHARDS) != 4 or len(SHARD_SHAS) != 4 or len(set(SHARDS)) != 4:
        raise ValueError("expected four distinct review files")
    sample = validated_sample(read_pinned(SAMPLE, SAMPLE_SHA))
    seen, selected, counts, sources = set(), set(), Counter(), []
    for path, expected in zip(SHARDS, SHARD_SHAS):
        data = json.loads(read_pinned(path, expected))
        inp = data.get("input", {}) or data.get("input_summary", {})
        if inp.get("sha256") != SAMPLE_SHA or inp.get("sample_file_sha256", SAMPLE_SHA) != SAMPLE_SHA:
            raise ValueError(f"review input digest mismatch: {path}")
        reviews = data.get("reviews") or data.get("items") or data.get("results")
        if not isinstance(reviews, list) or len(reviews) != SHARD_COUNT:
            raise ValueError(f"review count mismatch: {path}")
        for review in reviews:
            if not isinstance(review, dict) or set(review) != {"sample_id", "normalized_sha256", "label", "flags", "reviewer"}:
                raise ValueError(f"invalid review fields: {path}")
            sid = review["sample_id"]
            if not isinstance(sid, str) or sid not in sample or sid in seen:
                raise ValueError(f"duplicate or unknown review identity: {path}")
            if review["normalized_sha256"] != sample[sid]["normalized_sha256"]:
                raise ValueError(f"review identity digest mismatch: {path}")
            label = review["label"]
            if label not in {"clean", "minor", "major", "uncertain"} or not isinstance(review["flags"], list) or not isinstance(review["reviewer"], str) or not review["reviewer"]:
                raise ValueError(f"invalid review: {path}")
            seen.add(sid)
            counts[label] += 1
            if label == "clean":
                selected.add(sid)
        sources.append({"path": str(path), "sha256": expected, "records": len(reviews)})
    if seen != set(sample) or dict(counts) != LABEL_COUNTS or len(selected) != CLEAN_COUNT:
        raise ValueError("review coverage or label counts mismatch")

    # The blind queue is deliberately allowlisted, so no first-review fields leak.
    queue = b"".join((json.dumps({"sample_id": sid, "text": row["text"]}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for sid, row in sample.items() if sid in selected)
    summary = {
        "status": "second_blind_review_pending_not_training_approval",
        "source_sample": {"path": str(SAMPLE), "sha256": SAMPLE_SHA, "records": len(sample)},
        "source_reviews": sources,
        "review_queue": {"path": str(output / "samples.jsonl"), "sha256": sha(queue), "records": len(selected), "contains_text": True},
        "review_status": {"second_blind_review": "not_completed", "human_review": "not_completed", "training_approval": "not_generated"},
        "scope_controls": {"final_evaluation_read": False, "training_started": False},
    }
    output.mkdir(parents=False, exist_ok=False)
    (output / "samples.jsonl").write_bytes(queue)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
