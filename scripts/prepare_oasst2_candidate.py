"""Prepare an unapproved, tree-grouped OASST2 Chinese candidate.

The result is for local review only. No claim of semantic isolation or quality.
"""
import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import encode_dialogue, sha

SOURCE = ROOT / "runs/oasst2-chinese-audit-v1/2023-11-05_oasst2_ready.messages.jsonl.gz"
OLD = ROOT / "runs/oasst1-chinese-audit-v1/2023-04-12_oasst_ready.messages.jsonl.gz"
SOURCE_SHA = "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
OLD_SHA = "286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c"
RISK = re.compile(r"诊断|治疗|治療|剂量|用药|处方|自杀|自残|武器|炸弹|毒品|疫情|股票|投资|贷款|政治|政党|身份证|联系方式|住址|电话|ChatGPT|Open Assistant|开发者|训练数据|https?://|www\.|<\||\ufffd|```​?", re.I)
CONTACT = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def allowed(row):
    if row.get("lang") != "zh" or row.get("deleted") is True or row.get("review_result") is not True:
        return False
    if row.get("role") not in ("prompter", "assistant") or not isinstance(row.get("text"), str):
        return False
    text = row["text"].strip()
    if len(text) < 3 or len(text) > 2400 or RISK.search(text) or CONTACT.search(text):
        return False
    labels = row.get("labels") or {}
    required = ["pii", "spam", "lang_mismatch", "not_appropriate", "hate_speech", "sexual_content", "quality"]
    if row["role"] == "assistant":
        required.append("helpfulness")
    if any(not isinstance(labels.get(k), dict) or not labels[k].get("count") or labels[k].get("value") is None for k in required):
        return False
    if any(labels[k]["value"] > 0 for k in ("pii", "spam", "lang_mismatch")):
        return False
    if any(labels[k]["value"] > 0.25 for k in ("not_appropriate", "hate_speech", "sexual_content")):
        return False
    return labels["quality"]["value"] >= 0.5 and (row["role"] != "assistant" or labels["helpfulness"]["value"] >= 0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha(SOURCE) != SOURCE_SHA or sha(OLD) != OLD_SHA:
        raise ValueError("Pinned source changed")
    old_ids = set()
    with gzip.open(OLD, "rt", encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("lang") == "zh":
                old_ids.add(row["message_id"])
    dev_path = ROOT / "data/conversation-development.json"
    cases = json.loads(dev_path.read_text(encoding="utf8"))
    dev_prompts = [normalize(q) for c in cases["scenarios"] for q in c["turns"] if len(normalize(q)) >= 12]
    vocab = ROOT / "runs/conversation-data-v5/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(vocab))
    rows = {}
    counts = collections.Counter()
    with gzip.open(SOURCE, "rt", encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("lang") != "zh":
                continue
            counts["zh_rows"] += 1
            if allowed(row):
                rows[row["message_id"]] = row
    children = collections.defaultdict(list)
    for row in rows.values():
        children[row.get("parent_id")].append(row)
    candidates = []
    for row in rows.values():
        if row["role"] != "assistant" or row["message_id"] in children:
            continue
        path, visited, current = [], set(), row
        while current is not None and current["message_id"] not in visited:
            visited.add(current["message_id"])
            path.append(current)
            parent = current.get("parent_id")
            current = rows.get(parent) if parent else None
        path.reverse()
        if not path or path[0].get("parent_id") is not None or path[0]["role"] != "prompter" or len(path) % 2 or len(path) > 8:
            continue
        if any(path[i]["role"] == path[i + 1]["role"] for i in range(len(path) - 1)):
            continue
        if any(item["message_id"] in old_ids for item in path):
            counts["old_message_paths"] += 1
            continue
        messages = [{"role": "user" if item["role"] == "prompter" else "assistant", "content": item["text"].strip()} for item in path]
        if any(q in normalize(m["content"]) for m in messages for q in dev_prompts):
            counts["development_prompt_overlap"] += 1
            continue
        encoded = encode_dialogue(tokenizer, messages, limit=512)
        if encoded is None:
            counts["overlength"] += 1
            continue
        group = path[0]["message_tree_id"]
        candidates.append((group, path, messages, encoded))
    # One complete path per tree; deterministic ordering avoids branch leakage.
    seen = set()
    args.output.mkdir(parents=True)
    streams = {s: (args.output / (s + ".jsonl")).open("x", encoding="utf8") for s in ("train", "development")}
    try:
        for group, path, messages, (ids, mask) in sorted(candidates, key=lambda x: (x[0], tuple(r["message_id"] for r in x[1]))):
            if group in seen:
                counts["other_tree_branches"] += 1
                continue
            seen.add(group)
            bucket = int(hashlib.sha256(group.encode("utf8")).hexdigest()[:8], 16) % 10
            split = "development" if bucket == 0 else "train"
            item = {"group": group, "message_ids": [r["message_id"] for r in path], "messages": messages,
                    "tokens": ids, "mask": mask}
            streams[split].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[split + "_paths"] += 1
            counts[split + "_multiturn"] += len(messages) >= 4
            counts[split + "_targets"] += sum(mask)
    finally:
        for f in streams.values():
            f.close()
    manifest = {"status": "candidate_not_approved_for_training", "source_sha256": SOURCE_SHA,
                "old_source_sha256": OLD_SHA, "tokenizer_sha256": sha(vocab),
                "development_cases_sha256": sha(dev_path), "generator_sha256": sha(Path(__file__)),
                "counts": dict(counts), "files": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
                "limitations": ["Review labels and regex cannot establish factual correctness or privacy.",
                                "OASST2 may overlap previous datasets beyond old message IDs.",
                                "No semantic near-duplicate or independent content review has been completed.",
                                "Groups split at source tree, not a final chat evaluation."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
