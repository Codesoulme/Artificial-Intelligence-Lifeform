"""Audit several separated FineWeb2 Chinese row groups via HTTP Range."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import urllib.request
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p128-fineweb2-cmn-multigroup-20261109"
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
PATH = "data/cmn_Hani/train/000_00000.parquet"
URL = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/{REVISION}/{PATH}"
SIZE = 4_841_624_662
ROW_GROUPS = (0, 400, 800, 1200, 1600)


class RangeFile(io.RawIOBase):
    def __init__(self, url, size): self.url, self.size, self.position, self.bytes_read = url, size, 0, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position
    def read(self, count=-1):
        if count < 0: count = self.size - self.position
        if count > 20_000_000: raise ValueError("range request exceeds audit limit")
        if count == 0: return b""
        end = min(self.size, self.position + count)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end-1}"})
        with urllib.request.urlopen(req, timeout=120) as response: data = response.read()
        if len(data) != end - self.position: raise ValueError("range response mismatch")
        self.position = end; self.bytes_read += len(data); return data


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def run():
    if OUT.exists(): raise FileExistsError(OUT)
    remote = RangeFile(URL, SIZE); parquet = pq.ParquetFile(remote)
    if max(ROW_GROUPS) >= parquet.metadata.num_row_groups: raise ValueError("row group outside shard")
    columns = ["text", "id", "dump", "url", "date", "language", "language_score", "language_script", "minhash_cluster_size"]
    rows = []
    for group in ROW_GROUPS:
        rows.extend(parquet.read_row_group(group, columns=columns).to_pylist())
    domains = Counter(); languages = Counter(); scripts = Counter(); pii = Counter(); groups = Counter(); lengths = []
    for row in rows:
        text = row.get("text") or ""; url = row.get("url") or ""
        domains[(urlparse(url).hostname or "").lower()] += 1
        languages[str(row.get("language"))] += 1; scripts[str(row.get("language_script"))] += 1
        groups[str(row.get("minhash_cluster_size"))] += 1; lengths.append(len(text))
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text): pii["email_pattern"] += 1
        if re.search(r"(?:\+?86[- ]?)?1[3-9]\d{9}", text): pii["phone_pattern"] += 1
        if re.search(r"\b\d{17}[\dXx]\b", text): pii["id_pattern"] += 1
    OUT.mkdir(parents=False); sample = OUT / "sample.jsonl"
    with sample.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            text = row.get("text") or ""
            f.write(json.dumps({**row, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}, ensure_ascii=False, separators=(",", ":")) + "\n")
    result = {"status": "source_multigroup_audit_not_training", "revision": REVISION, "path": PATH,
              "row_groups": list(ROW_GROUPS), "rows": len(rows), "total_shard_rows": parquet.metadata.num_rows,
              "bytes_read_by_range": remote.bytes_read, "sample_sha256": sha(sample), "contains_text": True,
              "language": dict(languages), "script": dict(scripts), "domains_top30": domains.most_common(30),
              "cluster_size_top20": groups.most_common(20),
              "length": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths)/len(lengths)},
              "pii_pattern_counts": dict(pii),
              "limitations": ["Five row groups from one shard do not represent the full corpus.", "Regex matches are not proof of personal information.", "No semantic, rights, or cross-corpus review; no training."]}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text("# P128 FineWeb2 中文多 row group 审计\n\n" + f"从固定分片抽取 row group {list(ROW_GROUPS)}，共 {len(rows)} 条，HTTP Range 读取约 {remote.bytes_read/1024**2:.1f} MiB；正文只保存在被忽略的 runs 目录。\n\n" + "多组样本用于检查首组偏差，不构成全库质量、隐私或权利结论，也未生成训练候选。\n", encoding="utf-8")
    return result


if __name__ == "__main__": print(json.dumps(run(), ensure_ascii=False, indent=2))
