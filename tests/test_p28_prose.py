"""Regression checks for data integrity, split isolation and damaged prose filters."""
from collections import Counter
from array import array
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

from scripts.prepare_p28_prose import (
    HeldIndex, basic_topic_reasons, digest, extract, iter_json_array, normalized, output_dir,
    quality_reasons, signatures, split_of_group, ROOT, BASE, file_hash,
)


GOOD = ("松树在山坡上生长，枝叶在冬季仍然保持绿色，给周围的山地增添了色彩。"
        "松树的叶片细长，表面有一层保护组织，可以减少水分的散失。"
        "许多鸟类会在树枝间筑巢，把那里作为遮风避雨和抚育幼鸟的地方。")


class ProsePreparationTests(unittest.TestCase):
    def test_stream_parser_chunk_boundaries_and_source_lines(self):
        raw = '[\n {"completion":"春天\\n小河", "source":"甲"},\n {"completion":"夏天"}\n]\n'
        with patch.object(Path, "open", return_value=io.StringIO(raw)):
            rows = list(iter_json_array(Path("unused"), chunk_size=7))
        self.assertEqual([(r[0], r[1]) for r in rows], [(1, 2), (2, 3)])
        self.assertEqual(rows[0][2]["completion"], "春天\n小河")

    def test_parser_refuses_truncated_array_and_trailing_comma(self):
        for raw in ('[{"x":1}', '[{"x":1},]', '[{"x":1}] extra'):
            with self.subTest(raw=raw), patch.object(Path, "open", return_value=io.StringIO(raw)):
                with self.assertRaises(ValueError):
                    list(iter_json_array(Path("unused"), chunk_size=3))

    def test_extract_preserves_contiguous_original_offsets(self):
        article = "松树\n* 分类残片\n  " + GOOD + "  \n其他内容"
        selected = extract(article, Counter())
        self.assertIsNotNone(selected)
        text, start, end = selected
        self.assertEqual(text, GOOD)
        self.assertEqual(article[start:end], GOOD)

    def test_long_prefix_stops_at_complete_sentence_without_rewriting(self):
        extra = "远处的道路沿着山坡延伸，连接山下的村庄和附近的农田。" * 6
        article = GOOD + extra
        selected = extract(article, Counter(), max_chars=len(GOOD))
        self.assertEqual(selected[0], GOOD)
        self.assertEqual(article[selected[1]:selected[2]], GOOD)

    def test_no_splicing_short_lines_to_reach_minimum(self):
        self.assertIsNone(extract("小河流过村庄。\n" * 20, Counter()))

    def test_no_unfinished_or_context_dependent_passages(self):
        self.assertIn("unfinished_sentence", quality_reasons(GOOD[:-1]))
        self.assertIn("dependent_opening", quality_reasons("它们" + GOOD))
        self.assertIn("unbalanced_delimiter", quality_reasons("（" + GOOD))

    def test_empty_formula_markup_and_duplicate_sentence(self):
        self.assertIn("empty_formula_or_broken_punctuation", quality_reasons("定义  为  公式。" + GOOD))
        self.assertIn("markup_list_or_symbol", quality_reasons("thumb" + GOOD))
        self.assertIn("markup_list_or_symbol", quality_reasons("* " + GOOD))
        self.assertIn("repeated_sentence", quality_reasons(GOOD + GOOD))

    def test_normalized_article_group_stable_for_all_excerpts(self):
        original = "ＡＢＣ " + GOOD
        equivalent = "abc\n" + GOOD
        group = digest(normalized(original))
        self.assertEqual(group, digest(normalized(equivalent)))
        self.assertEqual(split_of_group(group), split_of_group(digest(normalized(equivalent))))
        self.assertEqual(split_of_group("0" * 64), "test")
        self.assertEqual(split_of_group(f"{25:016x}" + "0" * 48), "development")
        self.assertEqual(split_of_group(f"{50:016x}" + "0" * 48), "train")

    def test_held_exact_and_leading96_normalization(self):
        held = HeldIndex()
        text = GOOD + "小鸟在春季返回山林，寻找新的食物。"
        held.add(text)
        self.assertEqual(held.reason(" \n" + text), "held_exact")
        self.assertEqual(held.reason(normalized(text)[:96] + "新的内容。"), "held_leading96")

    def test_legacy_shingle_containment_excludes_nonprefix_overlap(self):
        text = "".join(chr(0x4e00 + i) for i in range(320))
        held = HeldIndex()
        held.add(text)
        prefix = "".join(chr(0x6000 + i) for i in range(16))
        self.assertEqual(held.reason(prefix + text[16:]), "held_sampled_lexical")
        self.assertIsNone(held.reason("完全无关的自然描述。"))
        self.assertEqual(len(signatures(text)), 20)

    def test_held_stored_prompt_group_also_excluded(self):
        held = HeldIndex()
        held.add("无关回答。", extra_group=digest(normalized(GOOD)[:96]))
        self.assertEqual(held.reason(GOOD), "held_leading96")

    def test_run_path_cannot_escape_new_output_root(self):
        for name in ("../old", "D:/old", "x/y", "x\\y"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                output_dir(name)

    def test_topic_gate_does_not_equate_full_sentences_with_daily_prose(self):
        self.assertEqual(basic_topic_reasons(GOOD), [])
        named = "某位科学家出生于一个村庄，后来担任研究人员。" + GOOD
        self.assertIn("named_catalog_or_specialist", basic_topic_reasons(named))
        chronology = "这一建筑在1935年建成，1978年又进行了修复。" + GOOD
        self.assertIn("named_catalog_or_specialist", basic_topic_reasons(chronology))
        self.assertIn("insufficient_common_description", basic_topic_reasons("文章句子完整，意思也很清楚。"))


@unittest.skipUnless(os.environ.get("P28_RUN"), "Set P28_RUN for generated-artifact checks")
class PreparedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = Path(os.environ["P28_RUN"]).resolve()
        if not cls.out.is_relative_to(BASE.resolve()):
            raise ValueError("Artifact test outside authorized output root")
        cls.manifest = json.loads((cls.out / "manifest.json").read_text("utf-8"))

    def test_all_artifact_hashes_and_source_unchanged(self):
        for filename, expected in self.manifest["files"].items():
            self.assertEqual(file_hash(self.out / filename), expected, filename)
        source = ROOT / "runs/chinese-prose-material" / self.manifest["source_provenance"]["path"]
        self.assertEqual(file_hash(source), self.manifest["source_sha256_verified"])
        self.assertEqual(file_hash(ROOT / "runs/conversation-data-v5/tokenizer.json"),
                         self.manifest["tokenizer_sha256"])

    def test_train_development_encoding_offsets_and_quality(self):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(self.out / "tokenizer.json"))
        bos, eos = [tokenizer.token_to_id(x) for x in ("<|bos|>", "<|eos|>")]
        all_groups = set()
        all_hashes = set()
        for split in ("train", "development"):
            offset = count = 0
            with (self.out / f"pretrain-{split}.jsonl").open(encoding="utf-8") as raw, \
                 (self.out / f"pretrain-{split}.bin").open("rb") as binary:
                for line in raw:
                    row = json.loads(line)
                    self.assertEqual(row["split"], split)
                    self.assertEqual(split_of_group(row["group"]), split)
                    self.assertNotIn(row["group"], all_groups)
                    self.assertNotIn(row["sha256"], all_hashes)
                    all_groups.add(row["group"]); all_hashes.add(row["sha256"])
                    self.assertEqual(quality_reasons(row["text"]), [])
                    self.assertEqual(basic_topic_reasons(row["text"]), [])
                    seq = [bos] + tokenizer.encode(row["text"], add_special_tokens=False).ids + [eos]
                    self.assertEqual(row["token_offset"], offset)
                    self.assertEqual(row["token_count"], len(seq))
                    packed = array("H", seq)
                    if sys.byteorder != "little":
                        packed.byteswap()
                    self.assertEqual(binary.read(len(seq) * 2), packed.tobytes())
                    offset += len(seq); count += 1
                self.assertEqual(binary.read(1), b"")
            self.assertEqual(offset, self.manifest["stats"][split]["tokens"])
            self.assertEqual(count, self.manifest["stats"][split]["documents"])

    def test_test_binary_only_boundaries_no_text_open(self):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(self.out / "tokenizer.json"))
        bos, eos = [tokenizer.token_to_id(x) for x in ("<|bos|>", "<|eos|>")]
        counts = Counter()
        with (self.out / "pretrain-test.bin").open("rb") as stream:
            while data := stream.read(131072):
                self.assertEqual(len(data) % 2, 0)
                seq = array("H")
                seq.frombytes(data)
                if sys.byteorder != "little":
                    seq.byteswap()
                counts["tokens"] += len(seq)
                counts["bos"] += seq.count(bos)
                counts["eos"] += seq.count(eos)
        self.assertEqual(counts["tokens"], self.manifest["stats"]["test"]["tokens"])
        self.assertEqual(counts["bos"], self.manifest["stats"]["test"]["documents"])
        self.assertEqual(counts["eos"], counts["bos"])

    def test_review_sample_original_spans_and_article_hashes(self):
        sample = json.loads((self.out / "quality-sample.json").read_text("utf-8"))["samples"]
        self.assertEqual(len(sample), 64)
        wanted = {r["source_row_1based"]: r for r in sample}
        source = ROOT / "runs/chinese-prose-material" / self.manifest["source_provenance"]["path"]
        for ordinal, line, article in iter_json_array(source):
            if ordinal not in wanted:
                continue
            row = wanted.pop(ordinal)
            original = article["completion"]
            self.assertEqual(row["source_line_1based"], line)
            self.assertEqual(row["source_original_sha256"], digest(original))
            self.assertEqual(row["group"], digest(normalized(original)))
            self.assertEqual(row["text"], original[row["source_char_start"]:row["source_char_end"]])
            self.assertEqual(row["split"], "train")
            if not wanted:
                break
        self.assertEqual(wanted, {})


if __name__ == "__main__":
    unittest.main()
