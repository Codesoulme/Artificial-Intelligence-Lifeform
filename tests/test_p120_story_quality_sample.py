import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import p120_story_quality_sample as sample


class StoryQualitySampleTests(unittest.TestCase):
    def test_allocation_is_proportional_and_deterministic(self):
        counts = {"000-127": 2, "128-255": 6, "256-383": 2, "384-511": 0, "512-plus": 0}
        first = sample.allocate(counts, 5)
        self.assertEqual(first, sample.allocate(counts, 5))
        self.assertEqual(sum(first.values()), 5)
        self.assertEqual(first["000-127"], 1)
        self.assertEqual(first["128-255"], 3)
        self.assertEqual(first["256-383"], 1)

    def test_selection_is_stable_by_group_and_source_row(self):
        db = sample._create_db()
        try:
            rows = [
                ("g1", "f1", "o1", 10, 1, 100, "000-127"),
                ("g2", "f2", "o2", 20, 2, 100, "000-127"),
                ("g3", "f3", "o3", 30, 3, 200, "128-255"),
            ]
            for group, full, original, source_row, line, tokens, stratum in rows:
                priority = sample.stable_key(sample.SEED, stratum, source_row, group, full)
                db.execute("INSERT INTO groups VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (group, full, original, source_row, line, tokens, stratum, priority))
            db.commit()
            selected = sample._selected_rows(
                db, {"000-127": 1, "128-255": 1, "256-383": 0, "384-511": 0, "512-plus": 0}, sample.SEED
            )
            self.assertEqual([row["group_sha256"] for row in selected],
                             [min(("g1", "g2"), key=lambda g: sample.stable_key(sample.SEED, "000-127", int(g[1]) * 10, g, "f" + g[1])), "g3"])
            self.assertEqual(len({row["group_sha256"] for row in selected}), 2)
        finally:
            db.close()

    def test_synthetic_run_keeps_body_out_of_summary_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "stories.jsonl"
            output = root / "runs" / "sample"
            output.parent.mkdir()
            texts = ["甲乙丙" * 40, "故事" * 90, "长度" * 140, "尾部" * 200]
            rows = []
            for line, text in enumerate(texts, 1):
                full, group = sample.normalized_identity(text)
                rows.append({
                    "text": text,
                    "source_row_1based": line,
                    "sha256": full,
                    "group": group,
                    "source_original_sha256": sample.digest(text),
                    "content_tokens": [100, 200, 300, 500][line - 1],
                    "input_line_1based": line,
                })
            input_sha = "a" * 64
            for row in rows:
                row["input_sha256"] = input_sha
            source.write_bytes(("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)).encode())
            old_root = sample.ROOT
            sample.ROOT = root
            self.addCleanup(lambda: setattr(sample, "ROOT", old_root))
            result = sample.run(source, output, expected_records=4, sample_size=4)
            summary_text = (output / "summary.json").read_text(encoding="utf-8")
            report_text = (output / "report.md").read_text(encoding="utf-8")
            self.assertNotIn("甲乙丙", summary_text)
            self.assertNotIn("甲乙丙", report_text)
            self.assertEqual(result["selected_count"], 4)
            self.assertEqual(sum(item["selected"] for item in result["strata"].values()), 4)
            sample_text = (output / "samples.jsonl").read_text(encoding="utf-8")
            self.assertIn("甲乙丙", sample_text)
            self.assertTrue(all("sample_id" in json.loads(line) for line in sample_text.splitlines()))
            self.assertEqual(result["scope_controls"]["final_evaluation_read"], False)
            self.assertEqual(result["review_status"]["training_approval"], "not_generated")

    def test_duplicate_source_row_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stories.jsonl"
            text = "重复" * 50
            full, group = sample.normalized_identity(text)
            rows = []
            for line in (1, 2):
                rows.append({"text": text + str(line), "source_row_1based": 1, "sha256": full,
                             "group": group, "source_original_sha256": sample.digest(text + str(line)),
                             "content_tokens": 100, "input_line_1based": line, "input_sha256": "x" * 64})
            source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sample.run(source, Path(directory) / "runs" / "sample", expected_records=2, sample_size=1)


if __name__ == "__main__":
    unittest.main()
