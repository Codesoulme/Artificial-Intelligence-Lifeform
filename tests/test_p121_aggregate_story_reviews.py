import json
from pathlib import Path
import tempfile
import unittest

from scripts import p121_aggregate_story_reviews as aggregate


class AggregateTests(unittest.TestCase):
    def test_rejects_duplicate_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            old = aggregate.SHARDS
            aggregate.SHARDS = tuple(old[:1] * 4)
            try:
                with self.assertRaises(ValueError):
                    aggregate.run(out)
            finally:
                aggregate.SHARDS = old

    def test_real_shards_cover_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            result = aggregate.run(Path(directory) / "out")
            self.assertEqual(result["sample"]["records"], 200)
            self.assertEqual(sum(result["labels"].values()), 200)
            self.assertEqual(sum(result["reviewers"].values()), 200)
            self.assertEqual(result["status"], "blind_model_review_aggregate_not_training_approval")


if __name__ == "__main__":
    unittest.main()
