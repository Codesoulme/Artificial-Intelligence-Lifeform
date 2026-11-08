"""Compare token units on non-test Chinese documents; no training or test reads."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs/chinese-curriculum-v2/manifest.json"
TOKENIZER = ROOT / "runs/conversation-data-v5/tokenizer.json"
OUT = ROOT / "runs/p156-token-unit-comparison-20261109"
MANIFEST_SHA = "3b53475fa0808a63a8443f2de01a9cf60328b54d1dc21c11b0e233b84671e1f8"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(manifest_path: Path = MANIFEST, tokenizer_path: Path = TOKENIZER) -> dict:
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("manifest changed")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    totals = {name: {"documents": 0, "content_units": 0, "targets": 0, "lengths": Counter()}
              for name in ("byte", "codepoint", "bpe")}
    split_counts = Counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    for row in manifest["documents"]:
        # The sealed test split is deliberately never opened by this diagnostic.
        if row["split"] not in ("train", "development"):
            continue
        split_counts[row["split"]] += 1
        text = (root / row["path"]).read_text(encoding="utf-8-sig")
        units = {
            "byte": len(text.encode("utf-8")),
            "codepoint": len(text),
            "bpe": len(tokenizer.encode(text, add_special_tokens=False).ids),
        }
        for name, count in units.items():
            item = totals[name]
            item["documents"] += 1
            item["content_units"] += count
            item["targets"] += count + 1  # BOS/EOS boundary target convention.
            item["lengths"][str(min(1024, count // 128 * 128))] += 1
    for item in totals.values():
        item["lengths"] = dict(sorted(item["lengths"].items(), key=lambda pair: int(pair[0])))
    return {"status": "representation_diagnostic_not_training", "manifest_sha256": MANIFEST_SHA,
            "tokenizer_sha256": sha(tokenizer_path), "splits_read": dict(split_counts),
            "test_split_read": False, "units": totals}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=False)
    result = summarize()
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
