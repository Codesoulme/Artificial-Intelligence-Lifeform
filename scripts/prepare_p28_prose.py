"""Stream existing attributed prose, mechanically isolate splits, encode unchanged BPE.

No model imports, downloads or training. Default prepares 20k candidates. Increase
--limit in a fresh --name directory under runs/p28-prose/. Never overwrite data.
Run --finalize NAME after writing that run's manual-review.json for the fixed
64-item TRAIN sample. Only this script's JSON report is updated on finalization.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import sys
import unicodedata

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
# Restrict this preparation process to at most two available logical CPUs on
# Windows as well as disabling library worker pools. No other process is touched.
if os.name == "nt":
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    current = kernel.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if kernel.GetProcessAffinityMask(current, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        bits = [1 << i for i in range(64) if process_mask.value & (1 << i)][:2]
        if bits and not kernel.SetProcessAffinityMask(current, sum(bits)):
            raise OSError("Could not limit preparation CPU affinity")
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/p28-prose"
REPORT = ROOT / "reports/P28-prose-preparation.json"
SEED = 20261109
SPLITS = ("train", "development", "test")


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def signatures(text):
    text = normalized(text)
    return {hashlib.blake2b(text[i:i + 16].encode(), digest_size=8).digest()
            for i in range(0, max(0, len(text) - 15), 16)}


def split_of_group(group):
    bucket = int(group[:16], 16) % 1000
    return "test" if bucket < 20 else "development" if bucket < 40 else "train"


def iter_json_array(path, chunk_size=65536):
    """Incremental array parser; at most one object plus a chunk is resident.

    Yields 1-based object ordinal, physical starting line, decoded object.
    A hard object bound rejects unexpectedly huge/malformed inputs.
    """
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer = ""
        line = 1
        eof = False

        def fill():
            nonlocal buffer, eof
            piece = stream.read(chunk_size)
            buffer += piece
            eof = not piece
            if len(buffer) > 8 * 1024 * 1024:
                raise ValueError("JSON object exceeds 8 MiB character bound")

        def consume(n):
            nonlocal buffer, line
            line += buffer[:n].count("\n")
            buffer = buffer[n:]

        def whitespace():
            while True:
                n = len(buffer) - len(buffer.lstrip())
                consume(n)
                if buffer or eof:
                    return
                fill()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("Expected source JSON array")
        consume(1)
        ordinal = 0
        after_comma = False
        while True:
            whitespace()
            if buffer.startswith("]"):
                if after_comma:
                    raise ValueError("Trailing comma")
                consume(1)
                whitespace()
                if buffer:
                    raise ValueError("Trailing JSON content")
                return
            start_line = line
            while True:
                try:
                    row, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("Malformed or incomplete JSON array") from None
                    fill()
            if not isinstance(row, dict):
                raise ValueError("Source record is not an object")
            consume(end)
            ordinal += 1
            yield ordinal, start_line, row
            whitespace()
            if buffer.startswith(","):
                consume(1)
                after_comma = True
            elif buffer.startswith("]"):
                after_comma = False
            else:
                raise ValueError("Missing array delimiter")


ARTIFACT = re.compile(
    r"thumb|\{\{|\}\}|\[\[|\]\]|<[^>]*>|https?://|www\.|\ufffd|"
    r"(?:^|\n)\s*(?:[*#|•]|\d+[.、])|[\\^_=]|\[\d+(?:[,，-]\d+)*\]|"
    r"(?:ISBN|ISSN|Category:|File:|formula|displaystyle)", re.I)
BAD_GAP = re.compile(r"[\u4e00-\u9fff]\s{2,}[\u4e00-\u9fff]|"
                     r"[，。；：][，。；：]|[（(]\s*[）)]|"
                     r"(?:当且仅当|定义为|表示为|等于|设为)\s*[，。；]|"
                     r"[\u4e00-\u9fff]\d{1,3}[，。；]|(?:公式|方程|定理|矩阵|量词|积分|微分)")
DEPENDENT = re.compile(r"^(?:他[们]?|她[们]?|它[们]?|其[中后]?|这[些种个一]|该|此[外时后]?|"
                       r"上述|然而|因此|所以|但是|此外|同时|后来|随后|另外|于是|其中|目前|至今|现在)")
END = re.compile(r"[。！？][”’」』）)]*$")
SENTENCES = re.compile(r"[^。！？]*[。！？][”’」』）)]*")
COMMON_TERMS = re.compile("|".join((
    "雨水", "水分", "土壤", "植物", "叶片", "树叶", "根部", "种子", "果实", "花朵",
    "动物", "昆虫", "鸟类", "鱼类", "生长", "食物", "栖息", "温度", "水蒸气", "云层",
    "空气", "阳光", "水流", "岩石", "光线", "声音", "太阳", "月亮", "海洋", "河流",
    "森林", "草原", "生物", "呼吸", "厨房", "食材", "食品", "食用", "蔬菜", "水果",
    "面粉", "淀粉", "洗涤", "衣服", "房屋", "道路", "交通", "家庭", "建筑", "材料",
    "纤维", "木材", "玻璃", "金属", "纸张", "燃烧", "热量", "加热", "冷却", "蒸发",
    "融化", "冻结", "泥土", "沙子", "水滴", "降雨", "降水", "气候", "季节", "冬季",
    "夏季", "秋季", "春季", "种植", "栽培", "收获", "味道", "气味", "颜色", "形状",
    "羽毛", "皮肤", "身体", "眼睛", "耳朵", "牙齿", "肌肉", "骨骼", "睡眠", "运动",
    "表面", "吸收", "储存", "厨房", "饮用", "清洗", "烹调", "烹饪", "煮熟", "蒸煮",
)))
SPECIALIST = re.compile(
    r"出生|生于|逝世|毕业|担任|任职|总统|皇帝|议会|选举|政党|战争|部队|冠军|联赛|"
    r"足球|俱乐部|球员|专辑|电影|电视剧|演员|导演|漫画|小说|角色|发行|上映|乐队|"
    r"行政区|自治|人口|省份|县城|市区|位于|坐落|海拔|平方公里|公顷|面积|"
    r"朝代|世纪|考古|王朝|教会|佛教|基督|宗教|神话|传说|仪式|信仰|"
    r"化合物|催化剂|聚合物|蛋白质|细胞|分子|离子|原子|量子|电子|电路|基因|"
    r"疾病|药物|治疗|手术|癌|患者|化学式|算法|函数|数论|拓扑|经济学|哲学|学说|"
    r"\d{3,4}年|\d+世纪|[·•]|[A-Za-z]{3,}")


def basic_topic_reasons(text):
    reasons = []
    if len(set(COMMON_TERMS.findall(text))) < 3:
        reasons.append("insufficient_common_description")
    if SPECIALIST.search(text):
        reasons.append("named_catalog_or_specialist")
    if len(re.findall(r"\d", text)) / max(1, len(text)) > .02:
        reasons.append("basic_numeric_dense")
    return reasons


def quality_reasons(text, min_chars=80, max_chars=360):
    reasons = []
    if len(text) < min_chars:
        return ["too_short"]
    if len(text) > max_chars:
        return ["too_long"]
    if not END.search(text):
        reasons.append("unfinished_sentence")
    if len(re.findall(r"[。！？]", text)) < 2:
        reasons.append("fewer_than_two_sentences")
    if ARTIFACT.search(text):
        reasons.append("markup_list_or_symbol")
    if BAD_GAP.search(text):
        reasons.append("empty_formula_or_broken_punctuation")
    if DEPENDENT.search(text):
        reasons.append("dependent_opening")
    if re.search(r"[\x00-\x08\x0b-\x1f]", text):
        reasons.append("control_character")
    n = max(1, len(text))
    if len(re.findall(r"[\u4e00-\u9fff]", text)) / n < .70:
        reasons.append("low_chinese_ratio")
    if len(re.findall(r"[A-Za-z]", text)) / n > .08:
        reasons.append("latin_dense")
    if len(re.findall(r"\d", text)) / n > .06:
        reasons.append("numeric_dense")
    if len(re.findall(r"\d{3,4}年", text)) >= 3:
        reasons.append("date_catalog")
    if re.search(r"(.)\1{3,}|(.{2,20})\2{2,}", text):
        reasons.append("repeated_run")
    sentences = [normalized(s) for s in re.split(r"[。！？]", text) if len(s.strip()) >= 8]
    if len(sentences) != len(set(sentences)):
        reasons.append("repeated_sentence")
    grams = [text[i:i+8] for i in range(max(0, len(text)-7))]
    if grams and 1 - len(set(grams)) / len(grams) > .16:
        reasons.append("repeated_8gram")
    for left, right in (("（", "）"), ("(", ")"), ("“", "”"), ("「", "」"), ("《", "》")):
        balance = 0
        bad = False
        for char in text:
            balance += (char == left) - (char == right)
            if balance < 0:
                bad = True
        if balance or bad:
            reasons.append("unbalanced_delimiter")
            break
    return sorted(set(reasons))


def extract(article, stats, min_chars=80, max_chars=360):
    """One unchanged contiguous paragraph/prefix of full sentences per article.

    Prefer the first eligible paragraph, skipping titles and damaged blocks.
    Never splice sentences across paragraphs; leading context cannot be repaired.
    """
    for part in re.finditer(r"[^\r\n]+", article):
        raw = part.group()
        text = raw.strip()
        start = part.start() + len(raw) - len(raw.lstrip())
        stats["paragraphs_considered"] += 1
        if len(text) > max_chars:
            ends = [m.end() for m in SENTENCES.finditer(text) if m.end() <= max_chars]
            if ends:
                text = text[:max(ends)]
                stats["long_paragraph_sentence_prefix"] += 1
        if len(text) < min_chars:
            reasons = ["too_short"]
        elif len(text) > max_chars:
            reasons = ["too_long"]
        else:
            reasons = basic_topic_reasons(text)
            if not reasons:
                reasons = quality_reasons(text, min_chars, max_chars)
        if reasons:
            stats["paragraphs_rejected"] += 1
            stats["paragraph_primary_" + reasons[0]] += 1
            for reason in reasons:
                stats["paragraph_reason_" + reason] += 1
            continue
        assert article[start:start + len(text)] == text
        return text, start, start + len(text)
    return None


class HeldIndex:
    """Legacy exact/leading96 and stride16 shingle containment. No text logging."""
    def __init__(self):
        self.exact = set()
        self.groups = set()
        self.sizes = []
        self.index = defaultdict(list)

    def add(self, text, extra_group=None, extra_digest=None):
        key = normalized(text)
        self.exact.add(digest(key))
        self.groups.add(digest(key[:96]))
        if extra_group:
            self.groups.add(extra_group)
        if extra_digest:
            self.exact.add(extra_digest)
        sigs = signatures(text)
        number = len(self.sizes)
        self.sizes.append(len(sigs))
        for sig in sigs:
            self.index[sig].append(number)

    def reason(self, text):
        key = normalized(text)
        if digest(key) in self.exact:
            return "held_exact"
        if digest(key[:96]) in self.groups:
            return "held_leading96"
        sigs = signatures(text)
        shared = Counter(i for sig in sigs for i in self.index.get(sig, ()))
        if any(min(len(sigs), self.sizes[i]) >= 8 and
               count / min(len(sigs), self.sizes[i]) >= .8 for i, count in shared.items()):
            return "held_sampled_lexical"
        return None


def load_old_held():
    index = HeldIndex()
    evidence = {}
    for version in ("v3", "v5"):
        directory = ROOT / f"runs/conversation-data-{version}"
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        evidence[str(directory / "manifest.json")] = {"sha256": file_hash(directory / "manifest.json")}
        for split in ("development", "test"):
            for kind in ("pretrain", "dialogue"):
                name = f"{kind}-{split}.jsonl"
                path = directory / name
                actual = file_hash(path)
                if actual != manifest["files"][name]:
                    raise ValueError("Old held file hash mismatch (content suppressed)")
                count = 0
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        index.add(row["text"], row.get("group"), row.get("sha256"))
                        if row.get("messages"):
                            for message in row["messages"]:
                                index.add(message["content"])
                        count += 1
                evidence[str(path)] = {"sha256": actual, "records": count}
    return index, evidence


def write_json(path, obj, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def output_dir(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Run name must be a simple directory name")
    return BASE / name


def prepare(args):
    from tokenizers import Tokenizer

    out = output_dir(args.name)
    if out.exists():
        raise FileExistsError("Run directory exists; use a fresh --name")
    material = ROOT / "runs/chinese-prose-material"
    provenance = json.loads((material / "provenance.json").read_text("utf-8"))
    source = material / provenance["path"]
    actual_source_hash = file_hash(source)
    if actual_source_hash != provenance["sha256"]:
        raise ValueError("Source does not match existing provenance")
    token_source = ROOT / "runs/conversation-data-v5/tokenizer.json"
    token_manifest = json.loads((token_source.parent / "manifest.json").read_text("utf-8"))
    token_hash = file_hash(token_source)
    if token_hash != token_manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(token_source))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    bos, eos = [tokenizer.token_to_id(s) for s in ("<|bos|>", "<|eos|>")]
    if bos is None or eos is None or tokenizer.get_vocab_size() > 65536:
        raise ValueError("Tokenizer incompatible with uint16 or BOS/EOS")
    held, held_evidence = load_old_held()
    print("Verified source/wordpiece files; loaded old held signatures only.", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(token_source, out / "tokenizer.json")
    write_json(out / "source-provenance.json", provenance)
    db = sqlite3.connect(out / "selection.sqlite")
    db.execute("PRAGMA cache_size=-4096")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE candidates (group_hash TEXT PRIMARY KEY, segment_hash TEXT UNIQUE, score TEXT, split TEXT, payload TEXT)")
    stats = Counter()
    source_bucket_counts = Counter()
    try:
        for ordinal, line, row in iter_json_array(source):
            stats["articles_scanned"] += 1
            if ordinal % 25000 == 0:
                db.commit()
                print(f"Scanned {ordinal} articles; eligible {stats['articles_eligible']}", flush=True)
            article = row.get("completion")
            if not isinstance(article, str) or not article:
                stats["article_empty"] += 1
                continue
            found = extract(article, stats, args.min_chars, args.max_chars)
            if found is None:
                stats["article_no_clean_paragraph"] += 1
                continue
            text, start, end = found
            # Whole-article mechanical exclusion catches overlaps outside excerpt.
            # Do this after cheap rejection; no text failing extraction is kept.
            why = held.reason(article)
            if why:
                stats["article_" + why] += 1
                continue
            why = held.reason(text)
            if why:
                stats["segment_" + why] += 1
                continue
            group = digest(normalized(article))
            segment_hash = digest(normalized(text))
            record = {
                "text": text, "sha256": segment_hash, "group": group,
                "source_row_1based": ordinal, "source_line_1based": line,
                "source_original_sha256": digest(article),
                "source_normalized_sha256": group,
                "source_char_start": start, "source_char_end": end,
                "source": row.get("source"), "source_file_sha256": actual_source_hash,
                "source_repository": provenance["repository"],
                "source_revision": provenance["revision"],
                "license": provenance["license_declared"],
            }
            split = split_of_group(group)
            try:
                db.execute("INSERT INTO candidates VALUES (?,?,?,?,?)", (
                    group, segment_hash, digest(f"{SEED}:{group}"), split,
                    json.dumps(record, ensure_ascii=False)))
            except sqlite3.IntegrityError:
                stats["article_or_segment_exact_duplicate"] += 1
                continue
            stats["articles_eligible"] += 1
        db.commit()
        db.execute("CREATE INDEX candidate_score ON candidates(score)")
        # Cap is applied after scanning the entire source; no prefix bias.
        db.execute("CREATE TABLE selected AS SELECT * FROM candidates ORDER BY score LIMIT ?", (args.limit,))
        db.commit()
        selected_n = db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
        new_held = HeldIndex()
        new_test = HeldIndex()
        for split, payload in db.execute("SELECT split,payload FROM selected WHERE split != 'train'"):
            row = json.loads(payload)
            new_held.add(row["text"])
            if split == "test":
                new_test.add(row["text"])
        selection_fingerprint = hashlib.sha256()
        sample = []
        sample_rng = random.Random(SEED)
        train_seen = 0
        output_stats = {s: Counter() for s in SPLITS}
        inventories = {}
        with ExitStack() as stack:
            raw_streams = {s: stack.enter_context((out / f"pretrain-{s}.jsonl").open("x", encoding="utf-8")) for s in SPLITS}
            bins = {s: stack.enter_context((out / f"pretrain-{s}.bin").open("xb")) for s in SPLITS}
            # Hash while writing: generated test files are never reopened as text.
            for split, payload in db.execute("SELECT split,payload FROM selected ORDER BY score"):
                row = json.loads(payload)
                text = row["text"]
                why = new_held.reason(text) if split == "train" else new_test.reason(text) if split == "development" else None
                if why:
                    stats["new_split_excluded_" + split + "_" + why] += 1
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                if not ids or tokenizer.decode(ids) != text:
                    stats["tokenizer_roundtrip_rejected_" + split] += 1
                    continue
                seq = [bos] + ids + [eos]
                if len(seq) > args.max_tokens:
                    stats["token_length_rejected_" + split] += 1
                    continue
                row["token_offset"] = output_stats[split]["tokens"]
                row["token_count"] = len(seq)
                row["split"] = split
                packed = array("H", seq)
                if packed.itemsize != 2:
                    raise ValueError("Platform unsigned-short is not 16 bits")
                if sys.byteorder != "little":
                    packed.byteswap()
                bins[split].write(packed.tobytes())
                raw_streams[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                output_stats[split]["documents"] += 1
                output_stats[split]["tokens"] += len(seq)
                output_stats[split]["within_document_next_token_targets"] += len(seq) - 1
                output_stats[split]["characters"] += len(text)
                bucket = min(9, (row["source_row_1based"] - 1) * 10 // stats["articles_scanned"])
                source_bucket_counts[f"{split}_{bucket}"] += 1
                selection_fingerprint.update(f"{split}:{row['sha256']}\n".encode())
                if split == "train":
                    train_seen += 1
                    slot = train_seen - 1 if train_seen <= 64 else sample_rng.randrange(train_seen)
                    if slot < 64:
                        if train_seen <= 64:
                            sample.append(row)
                        else:
                            sample[slot] = row
        sample.sort(key=lambda x: x["source_row_1based"])
        for i, row in enumerate(sample):
            row["review_index"] = i
        write_json(out / "quality-sample.json", {
            "seed": SEED, "selection": "Uniform Algorithm-R reservoir of final TRAIN documents, sorted by source row for review.",
            "samples": sample,
        })
        for s in SPLITS:
            if (out / f"pretrain-{s}.bin").stat().st_size != output_stats[s]["tokens"] * 2:
                raise AssertionError("Binary size mismatch")
        # Byte-only file hashing is mechanical; held text is not rendered.
        for path in sorted(out.iterdir()):
            if path.suffix in (".json", ".jsonl", ".bin"):
                inventories[path.name] = file_hash(path)
        manifest = {
            "schema_version": 1, "stage": "prepared_pending_manual_review",
            "passed": False, "ready_for_training": False,
            "mechanical_checks_passed": True, "manual_review_passed": False,
            "run_directory": str(out), "source_provenance": provenance,
            "tokenizer_sha256": token_hash, "vocabulary": tokenizer.get_vocab_size(),
            "parameters": {"limit": args.limit, "min_chars": args.min_chars,
                           "max_chars": args.max_chars, "max_tokens": args.max_tokens, "seed": SEED},
            "source_sha256_verified": actual_source_hash,
            "preparation_stats": dict(stats), "stats": {s: dict(v) for s, v in output_stats.items()},
            "selected_before_encoding_and_new_split_exclusion": selected_n,
            "source_row_decile_counts": dict(source_bucket_counts),
            "selection_fingerprint": selection_fingerprint.hexdigest(),
            "sample_n": len(sample), "files": inventories,
            "split": "96/2/2 by first16 hex of normalized full original-article SHA256 modulo1000; one excerpt per article; exact normalized excerpt dedup globally before splitting; full-source seeded SHA256 rank then cap.",
            "binary_format": "Little-endian uint16 concatenation of BOS + unchanged tokenizer.encode(text, add_special_tokens=False) + EOS; per-record token_offset/token_count in JSONL. Training window policies remain trainer responsibility.",
            "extraction": "Unchanged contiguous original paragraph, or its complete-sentence prefix; surrounding whitespace omitted. No rewriting, fact edits or cross-paragraph concatenation. One eligible passage per article.",
            "topic_gate": "At least3 distinct common life/nature/material terms; reject explicit biography/year/catalog/specialist terms, foreign-name strings and numeric density>2%. This conservative lexical proxy loses useful prose and does not guarantee semantic suitability; fixed manual review must distinguish common description from named encyclopedia entries.",
            "old_held_evidence": held_evidence,
            "isolation": "Old v3/v5 development/test exact normalized text, stored group, computed leading96 and stride16 16-character shingles: >=8 denominator, >=80% min-size containment; additionally individual dialogue messages indexed. Checks original article and excerpt. New selected held excerpts mechanically exclude overlapping new train, test also excludes development. Finite lexical check, not semantic isolation proof.",
            "constraints": {"cpu_threads": 1, "gpu": False, "training": False,
                            "downloads": False, "full_source_loaded": False,
                            "final_chat_questions_opened": False,
                            "held_text_displayed": False},
            "memory": "64KiB source chunks, <=8MiB one-object bound; SQLite disk spool and 4MiB page cache, disk sort; bounded held signature index; only 64 train examples retained for review.",
            "filter_counts": "Sequential length, topic, structural gates; only reasons at the first rejecting gate are recorded, and reasons within that gate overlap. paragraph_primary_* sums to paragraphs_rejected. Articles may have rejected paragraphs before their selected excerpt; no rejected text logged.",
            "script_sha256": file_hash(Path(__file__)),
            "manual_gate": "At least64 fixed TRAIN samples, all individually annotated; zero damaged, dependent or unsuitable-for-basic-prose samples required for ready. One subjective review does not guarantee factual accuracy. On failure preserve candidates but do not train this artifact.",
        }
        write_json(out / "manifest.json", manifest)
        write_json(REPORT, {"current_run": str(out), "manifest": manifest,
                            "observations": ["Mechanical preparation finished; manual gate still pending."],
                            "recommendation": "Not ready for training until manual gate and relevant tests pass."}, exclusive=False)
        print(json.dumps({"run": str(out), "stats": manifest["stats"], "ready_for_training": False}), flush=True)
    finally:
        db.close()


def finalize(name):
    out = output_dir(name)
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    annotations = json.loads((out / "manual-review.json").read_text("utf-8"))
    sample = json.loads((out / "quality-sample.json").read_text("utf-8"))["samples"]
    for filename, expected in manifest["files"].items():
        if file_hash(out / filename) != expected:
            raise ValueError("Prepared artifact changed")
    if annotations["sample_sha256"] != file_hash(out / "quality-sample.json"):
        raise ValueError("Manual review sample mismatch")
    labels = annotations["annotations"]
    if len(sample) != 64 or len(labels) != 64:
        raise ValueError("Require exactly64 reviewed examples")
    counts = Counter()
    for row, label in zip(sample, labels):
        if row["source_row_1based"] != label["source_row_1based"] or row["sha256"] != label["sha256"]:
            raise ValueError("Annotation ordering/hash mismatch")
        if not label["note"]:
            raise ValueError("Each annotation needs a reason")
        for key in ("damaged", "context_dependent", "unsuitable_basic_prose", "factual_concern"):
            if type(label[key]) is not bool:
                raise ValueError("Review flags must be boolean")
            counts[key] += label[key]
    mechanical_ok = manifest["mechanical_checks_passed"]
    review_ok = not any(counts.values())
    test_result = json.loads((out / "test-results.json").read_text("utf-8"))
    tests_ok = test_result["passed"] is True and test_result["script_sha256"] == file_hash(Path(__file__))
    manifest.update(stage="review_complete", manual_review_passed=review_ok,
                    passed=mechanical_ok and review_ok and tests_ok,
                    ready_for_training=mechanical_ok and review_ok and tests_ok,
                    review_counts=dict(counts), tests_passed=tests_ok)
    manifest["files"]["manual-review.json"] = file_hash(out / "manual-review.json")
    manifest["files"]["test-results.json"] = file_hash(out / "test-results.json")
    write_json(out / "manifest.json", manifest, exclusive=False)
    write_json(REPORT, {
        "current_run": str(out), "manifest": manifest,
        "manual_review": annotations,
        "manual_statistics": {key: {"count": value, "n": 64, "percent": round(value * 100 / 64, 3)} for key, value in counts.items()},
        "observations": annotations["observations"],
        "inferences": annotations["inferences"],
        "recommendation": annotations["recommendation"],
        "limits": ["64-example single-reviewer development audit, not human blind review or a corpus-wide fact check.",
                   "No semantic-isolation proof; short/shifted/paraphrased overlaps may escape legacy shingles.",
                   "2023 snapshot may be dated. Article-level original URLs are absent upstream; collection provenance and original row hashes are retained.",
                   "No trainer integration or model evaluation performed. Ready gate applies to this artifact only."],
    }, exclusive=False)
    print(json.dumps({"ready_for_training": manifest["ready_for_training"], "counts": dict(counts)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="candidate-v3")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=360)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--finalize")
    args = parser.parse_args()
    if args.finalize:
        finalize(args.finalize)
    else:
        if args.limit < 64 or not 32 <= args.min_chars < args.max_chars <= 1000 or args.max_tokens < 32:
            parser.error("Invalid preparation limits")
        prepare(args)


if __name__ == "__main__":
    main()
