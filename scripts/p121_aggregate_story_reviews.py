"""Validate and aggregate body-free P120 blind review shards."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "runs/p120-story-quality-sample-20261109/samples.jsonl"
SAMPLE_SHA = "b29dd2db7ba4f0601d9e524300207daa4eaaaa720b0f0e66f28b79481e401213"
OUT = ROOT / "runs/p121-story-quality-review-aggregate-20261109"
SHARDS = tuple(ROOT / f"runs/p120-story-quality-review-agent{i}.json" for i in range(1, 5))
LABELS = {"clean", "minor", "major", "uncertain"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_sample():
    if sha(SAMPLE) != SAMPLE_SHA:
        raise ValueError("sample changed")
    rows = {}
    with SAMPLE.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if "text" not in row or not isinstance(row["text"], str):
                raise ValueError("sample text missing")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in rows:
                raise ValueError("sample identity invalid or duplicated")
            rows[sample_id] = {"line": line_no, "normalized": row.get("normalized_sha256")}
    if len(rows) != 200:
        raise ValueError("sample count mismatch")
    return rows


def run(output=OUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sample = load_sample()
    labels = Counter()
    flags = Counter()
    reviewers = Counter()
    seen = set()
    shards = []
    for index, path in enumerate(SHARDS, 1):
        data = json.loads(path.read_text(encoding="utf-8"))
        inp = data.get("input", {}) or data.get("input_summary", {})
        if inp.get("sha256") != SAMPLE_SHA or (inp.get("sample_file_sha256") is not None and inp.get("sample_file_sha256") != SAMPLE_SHA):
            raise ValueError(f"shard {index}: input digest mismatch")
        reviews = data.get("reviews") or data.get("items") or data.get("results")
        if not isinstance(reviews, list) or len(reviews) != 50:
            raise ValueError(f"shard {index}: review count mismatch")
        shard_ids = set()
        for review in reviews:
            if set(review) - {"sample_id", "normalized_sha256", "label", "flags", "reviewer"}:
                raise ValueError(f"shard {index}: unexpected review field")
            sample_id = review.get("sample_id")
            if sample_id not in sample or sample_id in seen or sample_id in shard_ids:
                raise ValueError(f"shard {index}: sample identity mismatch")
            if review.get("normalized_sha256") != sample[sample_id]["normalized"]:
                raise ValueError(f"shard {index}: normalized digest mismatch")
            if review.get("label") not in LABELS or not isinstance(review.get("flags"), list):
                raise ValueError(f"shard {index}: invalid label or flags")
            if not isinstance(review.get("reviewer"), str) or not review["reviewer"]:
                raise ValueError(f"shard {index}: reviewer missing")
            shard_ids.add(sample_id)
            seen.add(sample_id)
            labels[review["label"]] += 1
            flags.update(str(flag) for flag in review["flags"])
            reviewers[review["reviewer"]] += 1
        shards.append({"file": str(path), "sha256": sha(path), "records": len(reviews),
                       "line_range": inp.get("line_range") or inp.get("lines_1based") or inp.get("selected_lines"),
                       "reviewer": sorted({r["reviewer"] for r in reviews})})
    if seen != set(sample):
        raise ValueError("shards do not cover the sample exactly")
    output.mkdir(parents=False, exist_ok=False)
    result = {
        "status": "blind_model_review_aggregate_not_training_approval",
        "sample": {"path": str(SAMPLE), "sha256": SAMPLE_SHA, "records": len(sample)},
        "shards": shards,
        "labels": dict(sorted(labels.items())),
        "flag_counts": dict(sorted(flags.items())),
        "reviewers": dict(sorted(reviewers.items())),
        "limitations": [
            "Labels are independent model review, not human ground truth.",
            "A 200-record stratified sample cannot establish full-corpus quality or authorization.",
            "No review proves semantic isolation from held data or historical model exposure.",
            "The aggregate does not authorize training; candidate remains pending review.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# P121 故事候选盲审汇总", "",
        "四个独立模型审查分片覆盖 P120 的 200 篇样本，每条身份摘要与样本文件核对一致。正文、摘录和训练批准均未写入汇总。", "",
        f"- clean：{labels['clean']}；minor：{labels['minor']}；major：{labels['major']}；uncertain：{labels['uncertain']}。",
        f"- 输入样本 SHA-256：`{SAMPLE_SHA}`。", "",
        "这只是模型盲审，不是人工真值。样本中存在 major 或大量 minor 时，不能把全量候选直接投入训练；仍需按缺陷规则隔离、人工或独立复核，并重新绑定候选清单。该汇总不构成训练批准。", "",
        "缺陷类别计数和分片摘要见本地 `runs/p121-story-quality-review-aggregate-20261109/summary.json`。",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
