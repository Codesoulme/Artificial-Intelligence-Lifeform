"""Create a body-bearing, ignored queue of P120 records flagged by blind review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "runs/p120-story-quality-sample-20261109/samples.jsonl"
REVIEW_DIR = ROOT / "runs/p120-story-quality-review-"
OUT = ROOT / "runs/p122-story-quality-rereview-20261109"
SAMPLE_SHA = "b29dd2db7ba4f0601d9e524300207daa4eaaaa720b0f0e66f28b79481e401213"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(SAMPLE) != SAMPLE_SHA:
        raise ValueError("sample changed")
    labels = {}
    for shard in sorted(ROOT.glob("runs/p120-story-quality-review-agent*.json")):
        data = json.loads(shard.read_text(encoding="utf-8"))
        rows = data.get("reviews") or data.get("items") or data.get("results")
        if not rows:
            raise ValueError(f"empty shard: {shard}")
        for row in rows:
            if row["label"] in {"minor", "major", "uncertain"}:
                if row["sample_id"] in labels:
                    raise ValueError("duplicate review identity")
                labels[row["sample_id"]] = row
    selected = []
    with SAMPLE.open(encoding="utf-8") as source:
        for line_no, raw in enumerate(source, 1):
            row = json.loads(raw)
            review = labels.get(row["sample_id"])
            if review:
                selected.append({
                    "queue_line": len(selected) + 1,
                    "sample_line_1based": line_no,
                    "sample_id": row["sample_id"],
                    "normalized_sha256": row["normalized_sha256"],
                    "first_review_label": review["label"],
                    "first_review_flags": review.get("flags", []),
                    "text": row["text"],
                })
    if len(selected) != 119:
        raise ValueError(f"expected 119 flagged records, observed {len(selected)}")
    OUT.mkdir(parents=False)
    sample_path = OUT / "rereview.jsonl"
    with sample_path.open("x", encoding="utf-8", newline="\n") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "status": "pending_independent_rereview_not_training",
        "source_sample_sha256": SAMPLE_SHA,
        "records": len(selected),
        "first_review_labels": {label: sum(r["first_review_label"] == label for r in selected)
                                for label in ("minor", "major", "uncertain")},
        "queue_sha256": sha(sample_path),
        "contains_text": True,
        "final_evaluation_read": False,
        "training_started": False,
        "limitations": [
            "The first labels are independent model review, not human truth.",
            "The queue is a review aid and cannot authorize training.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
