"""Audit FineWeb2 Chinese shard metadata without downloading shard bodies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
DATASET = "HuggingFaceFW/fineweb-2"
BASE_API = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
BASE_FILE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
OUT = ROOT / "runs/p126-fineweb2-cmn-metadata-20261109"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as stream:
        return json.load(stream)


def get_bytes(url):
    with urllib.request.urlopen(url, timeout=60) as stream:
        return stream.read()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    train_url = BASE_API + "/data/cmn_Hani/train?recursive=false&expand=false&limit=400"
    test_url = BASE_API + "/data/cmn_Hani/test?recursive=false&expand=false&limit=400"
    train = get_json(train_url)
    test = get_json(test_url)
    files = [x for x in train + test if x.get("type") == "file" and x.get("path", "").endswith(".parquet")]
    if not files or any("size" not in x or "oid" not in x for x in files):
        raise ValueError("incomplete shard metadata")
    readme_url = BASE_FILE + "README.md"
    readme = get_bytes(readme_url)
    result = {
        "status": "metadata_audit_only_body_not_downloaded",
        "dataset": DATASET,
        "revision": REVISION,
        "config": "cmn_Hani",
        "shards": {"train": len([x for x in train if x.get("type") == "file"]),
                   "test": len([x for x in test if x.get("type") == "file"])},
        "parquet_files": len(files),
        "bytes": sum(x["size"] for x in files),
        "min_bytes": min(x["size"] for x in files),
        "max_bytes": max(x["size"] for x in files),
        "first_files": [{k: x.get(k) for k in ("path", "size", "oid", "xetHash")} for x in files[:5]],
        "readme": {"url": readme_url, "sha256": sha_bytes(readme), "bytes": len(readme)},
        "source_urls": [
            f"https://huggingface.co/datasets/{DATASET}",
            "https://arxiv.org/abs/2506.20920",
        ],
        "limitations": [
            "No parquet body or document text was downloaded.",
            "Metadata and data-card statements do not prove row-level quality, privacy, authorization, or semantic isolation.",
            "No training or candidate export was performed.",
        ],
    }
    OUT.mkdir(parents=False)
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(
        "# P126 FineWeb2 中文元数据审计\n\n"
        f"固定 revision `{REVISION}` 的 `cmn_Hani` 共发现 {result['parquet_files']} 个 Parquet 文件，合计约 {result['bytes'] / 1024**4:.2f} TiB；单文件约 {result['min_bytes'] / 1024**3:.2f}–{result['max_bytes'] / 1024**3:.2f} GiB。\n\n"
        "本次只读取 Hugging Face 文件树元数据和 README 摘要，没有下载正文，因此尚不能评价中文语体、隐私或实际质量，也没有生成训练候选。下一步只能在可控的小分片或官方行接口可用时做分层审计。\n",
        encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
