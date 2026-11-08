"""Build a deterministic, metadata-only stratified review sample for P115 stories.

The selected story bodies are written only to the ignored ``runs`` sample file.
All summaries and the report contain hashes, provenance, and counts, never body text.
This module uses only the standard library: it does not load a model, tokenizer, or
evaluation data, and it does not start training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs/p115-full-candidate-encoded-20261109-v1/stories.jsonl"
DEFAULT_OUTPUT = ROOT / "runs/p120-story-quality-sample-20261109"
SEED = 12020261109
EXPECTED_RECORDS = 199_993
SAMPLE_SIZE = 200

# The upper bound is inclusive.  P115 currently accepts content lengths 0..510;
# the final bucket makes the sampler fail-safe if a future candidate run is longer.
STRATA = (
    ("000-127", 0, 127),
    ("128-255", 128, 255),
    ("256-383", 256, 383),
    ("384-511", 384, 511),
    ("512-plus", 512, None),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_identity(text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()
    return digest(normalized), digest(normalized[:96])


def stratum_for(content_tokens: int) -> str:
    for name, low, high in STRATA:
        if content_tokens >= low and (high is None or content_tokens <= high):
            return name
    raise ValueError(f"content token length is out of range: {content_tokens}")


def stable_key(seed: int, stratum: str, source_row: int, group: str, full: str) -> str:
    value = f"{seed}\0{stratum}\0{source_row}\0{group}\0{full}"
    return digest(value)


def sample_id(seed: int, stratum: str, source_row: int, group: str, full: str) -> str:
    return stable_key(seed, stratum, source_row, group, full)


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate exactly total group units by proportional largest remainder."""
    available = [(name, counts.get(name, 0)) for name, _, _ in STRATA if counts.get(name, 0)]
    if not available:
        return {name: 0 for name, _, _ in STRATA}
    target = min(total, sum(count for _, count in available))
    result = {name: 0 for name, _, _ in STRATA}
    if target < len(available):
        for name, _ in sorted(available, key=lambda item: (-item[1], item[0]))[:target]:
            result[name] = 1
        return result

    # Give every non-empty stratum one unit, then divide the remainder among
    # the remaining groups.  Ties are resolved by the fixed stratum order.
    for name, _ in available:
        result[name] = 1
    remaining = target - len(available)
    capacity = {name: count - 1 for name, count in available}
    denominator = sum(capacity.values())
    if remaining and denominator:
        exact = {name: remaining * capacity[name] / denominator for name in capacity}
        for name in capacity:
            result[name] += min(capacity[name], int(exact[name]))
        left = target - sum(result.values())
        ranked = sorted(
            capacity,
            key=lambda name: (-(exact[name] - int(exact[name])),
                              next(i for i, item in enumerate(STRATA) if item[0] == name)),
        )
        while left:
            moved = False
            for name in ranked:
                if result[name] < counts[name]:
                    result[name] += 1
                    left -= 1
                    moved = True
                    if not left:
                        break
            if not moved:
                raise ValueError("sample allocation exceeded available groups")
    return result


def _require_hex(row: dict, key: str, line_number: int) -> str:
    value = row.get(key)
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"line {line_number}: invalid {key}")
    return value


def parse_row(raw: bytes, line_number: int, metadata_input_sha: str | None) -> dict:
    try:
        row = json.loads(raw.decode("utf-8-sig" if line_number == 1 else "utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"line {line_number}: invalid JSON/UTF-8") from exc
    if not isinstance(row, dict) or type(row.get("text")) is not str:
        raise ValueError(f"line {line_number}: missing string text")
    source_row = row.get("source_row_1based")
    if type(source_row) is not int or source_row < 1:
        raise ValueError(f"line {line_number}: invalid source row")
    if row.get("input_line_1based") != line_number:
        raise ValueError(f"line {line_number}: input line metadata mismatch")
    row_input_sha = row.get("input_sha256")
    if type(row_input_sha) is not str or len(row_input_sha) != 64 or any(c not in "0123456789abcdef" for c in row_input_sha):
        raise ValueError(f"line {line_number}: invalid input hash metadata")
    if metadata_input_sha is not None and row_input_sha != metadata_input_sha:
        raise ValueError(f"line {line_number}: input hash metadata mismatch")
    content_tokens = row.get("content_tokens")
    if type(content_tokens) is not int or content_tokens < 0:
        raise ValueError(f"line {line_number}: invalid content token length")
    full = _require_hex(row, "sha256", line_number)
    group = _require_hex(row, "group", line_number)
    original = _require_hex(row, "source_original_sha256", line_number)
    expected_full, expected_group = normalized_identity(row["text"])
    if (full, group, original) != (expected_full, expected_group, digest(row["text"])):
        raise ValueError(f"line {line_number}: identity metadata mismatch")
    return {
        "text": row["text"],
        "source_row": source_row,
        "input_line": line_number,
        "full": full,
        "group": group,
        "original": original,
        "content_tokens": content_tokens,
        "stratum": stratum_for(content_tokens),
        "input_sha": row_input_sha,
    }


def _create_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE source_rows (source_row INTEGER PRIMARY KEY, input_line INTEGER NOT NULL)")
    db.execute("CREATE TABLE full_hashes (full_hash TEXT PRIMARY KEY)")
    db.execute(
        """CREATE TABLE groups (
            group_hash TEXT PRIMARY KEY,
            full_hash TEXT NOT NULL,
            original_hash TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            input_line INTEGER NOT NULL,
            content_tokens INTEGER NOT NULL,
            stratum TEXT NOT NULL,
            priority TEXT NOT NULL
        )"""
    )
    return db


def _first_pass(path: Path, expected_records: int, seed: int) -> tuple[sqlite3.Connection, dict]:
    db = _create_db()
    raw_counts = Counter()
    records = 0
    metadata_input_sha = None
    try:
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise ValueError(f"line {line_number}: blank input line")
                row = parse_row(raw, line_number, metadata_input_sha)
                if metadata_input_sha is None:
                    metadata_input_sha = row["input_sha"]
                try:
                    db.execute("INSERT INTO source_rows VALUES (?, ?)", (row["source_row"], line_number))
                    db.execute("INSERT INTO full_hashes VALUES (?)", (row["full"],))
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f"line {line_number}: duplicate source row or normalized hash") from exc
                raw_counts[row["stratum"]] += 1
                priority = stable_key(seed, row["stratum"], row["source_row"], row["group"], row["full"])
                db.execute(
                    """INSERT INTO groups VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(group_hash) DO UPDATE SET
                         full_hash=excluded.full_hash,
                         original_hash=excluded.original_hash,
                         source_row=excluded.source_row,
                         input_line=excluded.input_line,
                         content_tokens=excluded.content_tokens,
                         stratum=excluded.stratum,
                         priority=excluded.priority
                       WHERE excluded.priority < groups.priority""",
                    (row["group"], row["full"], row["original"], row["source_row"],
                     line_number, row["content_tokens"], row["stratum"], priority),
                )
                records += 1
        if records != expected_records:
            raise ValueError(f"expected {expected_records} records, observed {records}")
        db.commit()
        group_counts = dict(db.execute("SELECT stratum, COUNT(*) FROM groups GROUP BY stratum"))
        summary = {
            "records": records,
            "raw_stratum_counts": {name: raw_counts.get(name, 0) for name, _, _ in STRATA},
            "unique_group_stratum_counts": {name: group_counts.get(name, 0) for name, _, _ in STRATA},
            "unique_groups": db.execute("SELECT COUNT(*) FROM groups").fetchone()[0],
            "p115_input_sha256": metadata_input_sha,
        }
        return db, summary
    except Exception:
        db.close()
        raise


def _selected_rows(db: sqlite3.Connection, allocations: dict[str, int], seed: int) -> list[dict]:
    selected = []
    for name, _, _ in STRATA:
        for row in db.execute(
            """SELECT full_hash, group_hash, original_hash, source_row, input_line,
                      content_tokens, stratum, priority
               FROM groups WHERE stratum=? ORDER BY priority, source_row, group_hash LIMIT ?""",
            (name, allocations.get(name, 0)),
        ):
            full, group, original, source_row, input_line, tokens, stratum, priority = row
            selected.append({
                "sample_id": sample_id(seed, stratum, source_row, group, full),
                "stratum": stratum,
                "source_row_1based": source_row,
                "input_line_1based": input_line,
                "group_sha256": group,
                "normalized_sha256": full,
                "source_original_sha256": original,
                "content_tokens": tokens,
                "rank_sha256": priority,
            })
    return selected


def _write_samples(path: Path, source: Path, selected: list[dict], metadata_input_sha: str) -> str:
    by_line = {row["input_line_1based"]: row for row in selected}
    found = set()
    with source.open("rb") as inp, path.open("x", encoding="utf-8", newline="\n") as out:
        for line_number, raw in enumerate(inp, 1):
            row = by_line.get(line_number)
            if row is None:
                continue
            parsed = parse_row(raw, line_number, metadata_input_sha)
            if parsed["source_row"] != row["source_row_1based"] or parsed["group"] != row["group_sha256"]:
                raise ValueError(f"selected identity mismatch at input line {line_number}")
            record = dict(row)
            record["text"] = parsed["text"]
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            found.add(line_number)
    if found != set(by_line):
        raise ValueError("selected input line missing during body extraction")
    return file_sha256(path)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _write_report(path: Path, summary: dict) -> None:
    counts = summary["strata"]
    lines = [
        "# P120 故事候选质量抽样报告",
        "",
        "本工具对 P115 训练侧候选做固定种子、按内容词元长度分层的组级稳定抽样。报告只记录来源、身份哈希和计数，不复制故事正文。",
        "",
        "## 运行边界",
        "",
        f"- 输入记录：{summary['source']['records']} 条；输入文件 SHA-256：`{summary['source']['sha256']}`。",
        f"- 固定种子：`{summary['seed']}`；目标抽样数：{summary['sample_size']}；实际抽样数：{summary['selected_count']}。",
        "- 抽样单位是组；组内按稳定哈希选择一条来源行。选择顺序不依赖 JSONL 行的遍历顺序。",
        "- 工具只读取指定的 P115 `stories.jsonl`，未读取最终评测数据；没有调用外部模型、没有加载模型、没有启动训练。",
        "",
        "## 分层计数",
        "",
        "| 分层 | 输入行数 | 唯一组数 | 抽样数 |",
        "|---|---:|---:|---:|",
    ]
    for name, _, _ in STRATA:
        item = counts[name]
        lines.append(f"| `{name}` | {item['input_records']} | {item['unique_groups']} | {item['selected']} |")
    lines += [
        "",
        "## 审查状态",
        "",
        "模型盲审尚未完成，人工审查尚未完成。抽样结果不构成质量结论，不生成训练批准；任何训练使用仍需单独完成审查、边界核验与批准流程。",
        "",
        "正文仅写入同一 runs 目录下被忽略的 `samples.jsonl`，汇总文件与本报告不含正文。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
        *, seed: int = SEED, sample_size: int = SAMPLE_SIZE,
        expected_records: int = EXPECTED_RECORDS) -> dict:
    source = Path(input_path).resolve(strict=True)
    output = Path(output_path).resolve()
    runs_root = (ROOT / "runs").resolve(strict=True)
    if not output.is_relative_to(runs_root) or output == runs_root:
        raise ValueError("output must be a new directory below runs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if sample_size < 1 or seed < 0 or expected_records < 1:
        raise ValueError("seed, expected_records, and sample_size must be positive")
    if source == output or output in source.parents:
        raise ValueError("output must not contain the input")

    candidate_file_sha = file_sha256(source)
    with tempfile.TemporaryDirectory(prefix="p120-story-sample-"):
        db, source_summary = _first_pass(source, expected_records, seed)
        try:
            allocations = allocate(source_summary["unique_group_stratum_counts"], sample_size)
            selected = _selected_rows(db, allocations, seed)
        finally:
            db.close()
    if len(selected) != min(sample_size, source_summary["unique_groups"]):
        raise ValueError("selected count does not match allocation")

    output.mkdir(parents=False, exist_ok=False)
    try:
        sample_path = output / "samples.jsonl"
        sample_sha = _write_samples(sample_path, source, selected, source_summary["p115_input_sha256"])
        if file_sha256(source) != candidate_file_sha:
            raise ValueError("input changed after sampling")
        selected_by_stratum = Counter(row["stratum"] for row in selected)
        strata = {}
        for name, _, _ in STRATA:
            strata[name] = {
                "input_records": source_summary["raw_stratum_counts"].get(name, 0),
                "unique_groups": source_summary["unique_group_stratum_counts"].get(name, 0),
                "selected": selected_by_stratum.get(name, 0),
            }
        summary = {
            "status": "sample_created_for_pending_review",
            "seed": seed,
            "sample_size": sample_size,
            "selected_count": len(selected),
            "source": {
                "path": str(source),
                "sha256": candidate_file_sha,
                "records": source_summary["records"],
                "unique_groups": source_summary["unique_groups"],
                "training_side": "caller_asserted_by_p115_input",
                "p115_input_sha256": source_summary["p115_input_sha256"],
            },
            "strata": strata,
            "identity_fields": [
                "sample_id", "normalized_sha256", "source_original_sha256",
                "group_sha256", "source_row_1based", "input_line_1based",
            ],
            "stable_order": "SHA256(seed NUL stratum NUL source_row_1based NUL group_sha256 NUL normalized_sha256)",
            "sample_file": {"path": str(sample_path), "sha256": sample_sha, "contains_text": True},
            "review_status": {
                "model_blind_review": "not_completed",
                "human_review": "not_completed",
                "training_approval": "not_generated",
            },
            "scope_controls": {
                "final_evaluation_read": False,
                "external_model_called": False,
                "training_started": False,
            },
            "limitations": [
                "Length strata use the already recorded P115 content_tokens field; no tokenizer is loaded.",
                "Counts describe this candidate file only and do not establish data quality, authorization, or evaluation isolation.",
            ],
        }
        _write_json(output / "summary.json", summary)
        _write_report(output / "report.md", summary)
        return summary
    except Exception as exc:
        # Keep failure metadata body-free and make the partial output explicitly unusable.
        _write_json(output / "failure.json", {
            "status": "failed_do_not_use_partial_outputs",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--expected-records", type=int, default=EXPECTED_RECORDS)
    args = parser.parse_args(argv)
    try:
        summary = run(args.input, args.output, sample_size=args.sample_size,
                      expected_records=args.expected_records)
    except Exception as exc:
        print(f"sampling failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
