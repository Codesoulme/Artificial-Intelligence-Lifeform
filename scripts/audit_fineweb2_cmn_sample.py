"""Range-read one FineWeb2 Chinese Parquet row group for local-only audit."""
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
OUT = ROOT / "runs/p127-fineweb2-cmn-sample-20261109"
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"
PATH = "data/cmn_Hani/train/000_00000.parquet"
URL = f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/{REVISION}/{PATH}"
SIZE = 4_841_624_662
OID = "4f89adce91622dcc35aab13dc808c311a662df59"


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.bytes_read = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if whence == 0: self.position = offset
        elif whence == 1: self.position += offset
        elif whence == 2: self.position = self.size + offset
        else: raise ValueError("invalid whence")
        return self.position

    def read(self, count=-1):
        if count < 0: count = self.size - self.position
        if count == 0: return b""
        if count > 20_000_000: raise ValueError("range request exceeds audit limit")
        end = min(self.size, self.position + count)
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end - 1}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if len(data) != end - self.position:
            raise ValueError("range response length mismatch")
        self.position = end
        self.bytes_read += len(data)
        return data


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def audit():
    if OUT.exists(): raise FileExistsError(OUT)
    remote = RangeFile(URL, SIZE)
    parquet = pq.ParquetFile(remote)
    if parquet.schema_arrow.names != ["text", "id", "dump", "url", "date", "file_path", "language", "language_score", "language_script", "minhash_cluster_size", "top_langs"]:
        raise ValueError("unexpected schema")
    table = parquet.read_row_group(0, columns=["text", "id", "dump", "url", "date", "language", "language_score", "language_script", "minhash_cluster_size"])
    rows = table.to_pylist()
    if len(rows) != 1000: raise ValueError("unexpected row group size")
    output_rows = []
    domains = Counter(); languages = Counter(); scripts = Counter(); pii = Counter(); clusters = Counter(); lengths = []
    for row in rows:
        text = row.get("text") or ""
        url = row.get("url") or ""
        domain = (urlparse(url).hostname or "").lower()
        domains[domain] += 1; languages[str(row.get("language"))] += 1; scripts[str(row.get("language_script"))] += 1
        clusters[str(row.get("minhash_cluster_size"))] += 1; lengths.append(len(text))
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text): pii["email_pattern"] += 1
        if re.search(r"(?:\+?86[- ]?)?1[3-9]\d{9}", text): pii["phone_pattern"] += 1
        if re.search(r"\b\d{17}[\dXx]\b", text): pii["id_pattern"] += 1
        output_rows.append({**row, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    OUT.mkdir(parents=False)
    sample = OUT / "sample.jsonl"
    with sample.open("x", encoding="utf-8", newline="\n") as f:
        for row in output_rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "status": "source_sample_audit_not_training",
        "source": {"revision": REVISION, "path": PATH, "url": URL, "size": SIZE, "oid": OID, "row_group": 0, "rows": len(rows)},
        "parquet": {"total_rows": parquet.metadata.num_rows, "row_groups": parquet.metadata.num_row_groups, "schema": parquet.schema_arrow.names},
        "sample": {"sha256": sha(sample), "contains_text": True, "bytes_read_by_range": remote.bytes_read},
        "language": dict(languages), "script": dict(scripts), "domains_top20": domains.most_common(20),
        "cluster_size_top20": clusters.most_common(20),
        "length": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "pii_pattern_counts": dict(pii),
        "limitations": ["One row group is not representative of the whole source.", "Regex matches are not proof of personal information; misses are not proof of safety.", "No semantic review, rights approval, cross-corpus isolation, or training was performed."],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text("# P127 FineWeb2 中文小分片审计\n\n" +
        f"固定 revision 的 `{PATH}` 第 0 个 row group 通过 HTTP Range 读取 {len(rows)} 条记录，读取网络字节约 {remote.bytes_read / 1024**2:.1f} MiB；正文样本只保存在被忽略的 runs 目录。\n\n"+
        "本次只证明该 row group 可被受控读取并提供 URL、日期、语言和去重簇字段；来源域、隐私命中和长度分布不能外推整包。未生成训练候选或批准。\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
