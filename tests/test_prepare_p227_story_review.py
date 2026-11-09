import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import p120_story_quality_sample as p120
from scripts import prepare_p227_story_review as review


class PrepareReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.runs.mkdir()
        self.sample = self.runs / "sample.jsonl"
        self.shards = [self.runs / f"review{i}.json" for i in range(4)]
        self.rows = []
        for i in range(4):
            text = f"Story {i} has distinct content."
            full, group = p120.normalized_identity(text)
            sid = p120.sample_id(review.SEED, "000-127", i + 1, group, full)
            self.rows.append({"text": text, "sample_id": sid, "rank_sha256": sid,
                              "normalized_sha256": full, "group_sha256": group,
                              "source_original_sha256": p120.digest(text),
                              "source_row_1based": i + 1, "input_line_1based": i + 1,
                              "content_tokens": 10, "stratum": "000-127"})
        self.write_sample()
        self.reviews = [{"sample_id": row["sample_id"], "normalized_sha256": row["normalized_sha256"],
                         "label": "clean" if i < 2 else "minor", "flags": [], "reviewer": "reviewer"}
                        for i, row in enumerate(self.rows)]
        self.write_shards()
        self.settings = patch.multiple(review, ROOT=self.root, SAMPLE=self.sample,
                                       SHARDS=tuple(self.shards), SAMPLE_COUNT=4, SHARD_COUNT=1,
                                       CLEAN_COUNT=2, LABEL_COUNTS={"clean": 2, "minor": 2})
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def write_sample(self):
        self.sample.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.sample_sha = review.sha(self.sample.read_bytes())

    def write_shards(self):
        for path, item in zip(self.shards, self.reviews):
            path.write_text(json.dumps({"input": {"sha256": self.sample_sha}, "reviews": [item]}), encoding="utf-8")

    def run_queue(self, name="out"):
        hashes = tuple(review.sha(path.read_bytes()) for path in self.shards)
        with patch.multiple(review, SAMPLE_SHA=self.sample_sha, SHARD_SHAS=hashes):
            return review.run(self.runs / name)

    def test_queue_contains_only_identity_and_original_text(self):
        result = self.run_queue()
        rows = [json.loads(line) for line in (self.runs / "out/samples.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [{"sample_id": r["sample_id"], "text": r["text"]} for r in self.rows[:2]])
        self.assertEqual(result["review_queue"]["sha256"], review.sha((self.runs / "out/samples.jsonl").read_bytes()))
        self.assertEqual(result["review_status"]["training_approval"], "not_generated")
        self.assertNotIn("Story 0", (self.runs / "out/summary.json").read_text(encoding="utf-8"))

    def test_modified_file_digest_is_rejected(self):
        with patch.multiple(review, SAMPLE_SHA="0" * 64, SHARD_SHAS=tuple(review.sha(p.read_bytes()) for p in self.shards)):
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                review.run(self.runs / "out")
        self.assertFalse((self.runs / "out").exists())

    def test_tampered_text_even_with_updated_file_digest_is_rejected(self):
        self.rows[0]["text"] = "altered"
        self.write_sample()
        self.write_shards()
        with self.assertRaisesRegex(ValueError, "text identity mismatch"):
            self.run_queue()

    def test_duplicate_sample_source_identity_is_rejected(self):
        self.rows[1]["input_line_1based"] = self.rows[0]["input_line_1based"]
        self.write_sample()
        self.write_shards()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_queue()

    def test_duplicate_review_and_missing_coverage_are_rejected(self):
        self.reviews[1] = self.reviews[0].copy()
        self.write_shards()
        with self.assertRaisesRegex(ValueError, "duplicate or unknown"):
            self.run_queue()
        self.assertFalse((self.runs / "out").exists())

    def test_review_digest_mismatch_is_rejected(self):
        self.reviews[0]["normalized_sha256"] = "0" * 64
        self.write_shards()
        with self.assertRaisesRegex(ValueError, "review identity digest"):
            self.run_queue()

    def test_output_boundary_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "below runs"):
            review.run(self.root / "outside")
        self.run_queue()
        with self.assertRaises(FileExistsError):
            self.run_queue()


if __name__ == "__main__":
    unittest.main()
