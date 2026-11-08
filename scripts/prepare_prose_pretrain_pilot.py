"""Prepare a mechanically screened continuous-Chinese pretraining pilot."""
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

import ijson
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "runs" / "chinese-prose-material"
SOURCE = SOURCE_DIR / "wikipedia-cn-20230720-filtered.json"
BASE = ROOT / "runs" / "conversation-data-v5"
OUT = ROOT / "runs" / "prose-pretrain-v1"
REVIEW = ROOT / "reports" / "P49-prose-pilot-review.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def usable(text):
    text = normalized(text)
    if not 160 <= len(text) <= 24000:
        return None
    if any(marker in text.lower() for marker in ("thumb", "{{", "}}", "http://", "https://", "isbn", "doi:")):
        return None
    if re.search(r"\[[0-9]{1,3}\]", text) or re.search(r"(.{2,24})\1{3,}", text):
        return None
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
    if cjk < 100 or cjk / max(1, len(text)) < 0.45:
        return None
    digits = sum(char.isdigit() for char in text)
    if digits / max(1, len(text)) > 0.16:
        return None
    return text


def split_for(key):
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "development" if bucket < 2 else ("test" if bucket < 4 else "train")


def main():
    if OUT.exists() or REVIEW.exists():
        raise FileExistsError("Pilot artifacts already exist")
    OUT.mkdir()
    tokenizer = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    split_names = ("train", "development", "test")
    json_streams = {name: (OUT / f"pretrain-{name}.jsonl").open("w", encoding="utf-8") for name in split_names}
    bin_streams = {name: (OUT / f"pretrain-{name}.bin").open("wb") for name in split_names}
    seen = set()
    rejected = {}
    stats = {"source_records": 0, "kept_records": 0, "rejected": rejected}
    for name in split_names:
        stats[f"{name}_records"] = 0
        stats[f"{name}_tokens"] = 0
    try:
        with SOURCE.open("rb") as source_stream:
            for row in ijson.items(source_stream, "item"):
                stats["source_records"] += 1
                if stats["source_records"] % 10000 == 0:
                    print(json.dumps({"source_records": stats["source_records"], "kept_records": stats["kept_records"]}), flush=True)
                text = usable(row.get("completion", ""))
                if text is None:
                    rejected["mechanical_filter"] = rejected.get("mechanical_filter", 0) + 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    rejected["exact_duplicate"] = rejected.get("exact_duplicate", 0) + 1
                    continue
                seen.add(digest)
                split = split_for(digest)
                item = {"text": text, "source": row.get("source", "wikipedia.zh2307"), "sha256": digest}
                json_streams[split].write(json.dumps(item, ensure_ascii=False) + "\n")
                ids = [tokenizer.token_to_id("<|bos|>")] + tokenizer.encode(text).ids + [tokenizer.token_to_id("<|eos|>")]
                np.asarray(ids, dtype=np.uint16).tofile(bin_streams[split])
                stats["kept_records"] += 1
                stats[f"{split}_records"] += 1
                stats[f"{split}_tokens"] += len(ids)
    finally:
        for stream in (*json_streams.values(), *bin_streams.values()):
            stream.close()

    for path in BASE.iterdir():
        if path.name.startswith("dialogue-") or path.name in {"focus-tokens.json", "tokenizer.json"}:
            shutil.copy2(path, OUT / path.name)
    shutil.copy2(BASE / "tokenizer.json", OUT / "tokenizer.json")
    manifest = {
        "status": "experimental_mechanical_screen_only",
        "source_file_sha256": sha(SOURCE),
        "source_provenance_sha256": sha(SOURCE_DIR / "provenance.json"),
        "tokenizer_sha256": sha(OUT / "tokenizer.json"),
        "stats": stats,
        "limitations": [
            "Mechanical filtering only; factual claims are not independently verified.",
            "Some mathematical, citation, and identifier residue may remain.",
            "Source is CC BY-SA; retain attribution and share-alike obligations.",
        ],
    }
    manifest["files"] = {path.name: sha(path) for path in OUT.iterdir() if path.is_file()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = {
        "passed": True,
        "scope": "Exact artifact binding for an internal mechanical-screen pilot; not a quality certification.",
        "manifest_sha256": {"runs/prose-pretrain-v1": sha(OUT / "manifest.json")},
        "data_artifact_sha256": manifest["files"],
        "issues": manifest["limitations"],
    }
    (OUT / "lineage-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    review = {
        "status": "mechanical_screen_only",
        "approved_for_pilot": False,
        "experimental_internal_use": True,
        "data_manifest_sha256": sha(OUT / "manifest.json"),
        "input_directory": "runs/prose-pretrain-v1",
        "stats": stats,
        "limitations": manifest["limitations"],
    }
    REVIEW.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
