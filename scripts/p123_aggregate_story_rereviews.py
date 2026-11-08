"""Validate and aggregate the four body-free P122 rereview shards."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "runs/p122-story-quality-rereview-20261109/rereview.jsonl"
QUEUE_SHA = "e72caed5891f8ecec3150e64a59ba8b8ad1c4528f33cc528e2e09e8358a8121b"
OUT = ROOT / "runs/p123-story-quality-rereview-aggregate-20261109"
SHARDS = tuple(ROOT / f"runs/p122-story-quality-rereview-agent{i}.json" for i in range(1, 5))
LABELS = {"keep", "hold", "reject", "uncertain"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha(QUEUE) != QUEUE_SHA:
        raise ValueError("queue changed")
    queue = {}
    with QUEUE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queue[row["queue_line"]] = row
    if set(queue) != set(range(1, 120)):
        raise ValueError("queue coverage mismatch")
    seen = set(); labels = Counter(); flags = Counter(); reviewers = Counter(); shard_info = []
    for index, path in enumerate(SHARDS, 1):
        data = json.loads(path.read_text(encoding="utf-8"))
        inp = data.get("input", {}) or data.get("input_summary", {})
        if inp.get("sha256") != QUEUE_SHA:
            raise ValueError(f"shard {index}: queue digest mismatch")
        rows = data.get("reviews") or data.get("items") or data.get("results")
        if not isinstance(rows, list) or len(rows) not in (29, 30):
            raise ValueError(f"shard {index}: count mismatch")
        for row in rows:
            qline = row.get("queue_line")
            if qline not in queue or qline in seen or row.get("sample_id") != queue[qline]["sample_id"]:
                raise ValueError(f"shard {index}: identity mismatch")
            if row.get("normalized_sha256") != queue[qline]["normalized_sha256"]:
                raise ValueError(f"shard {index}: normalized digest mismatch")
            if row.get("rereview_label") not in LABELS or not isinstance(row.get("flags"), list):
                raise ValueError(f"shard {index}: invalid label")
            seen.add(qline); labels[row["rereview_label"]] += 1; flags.update(map(str, row["flags"]))
            reviewers[row.get("reviewer", "unknown")] += 1
        shard_info.append({"path": str(path), "sha256": sha(path), "records": len(rows),
                           "line_range": inp.get("line_range") or inp.get("reviewed_line_range")})
    if seen != set(queue):
        raise ValueError("shards do not cover queue exactly")
    OUT.mkdir(parents=False, exist_ok=False)
    result = {
        "status": "rereview_aggregate_not_training_approval",
        "queue": {"path": str(QUEUE), "sha256": QUEUE_SHA, "records": len(queue)},
        "shards": shard_info, "labels": dict(sorted(labels.items())),
        "flag_counts": dict(sorted(flags.items())), "reviewers": dict(sorted(reviewers.items())),
        "limitations": [
            "Second labels are still model review, not human ground truth.",
            "The queue is a stratified sample of a first review, not a random estimate of the full corpus.",
            "Keep records are not approved for training without source, overlap and content review.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(
        "# P123 故事候选二次复核汇总\n\n"
        f"119 条首轮问题样本的二次模型复核结果：keep {labels['keep']}，hold {labels['hold']}，reject {labels['reject']}，uncertain {labels['uncertain']}。\n\n"
        "该结果仍不是人工真值；队列来自首轮问题样本，存在选择偏差。keep 只表示二次模型暂未发现阻断项，不能直接训练或证明全量候选质量。\n",
        encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
