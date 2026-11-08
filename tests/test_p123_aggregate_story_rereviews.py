from pathlib import Path
import tempfile
import unittest

from scripts import p123_aggregate_story_rereviews as aggregate


class AggregateRereviewTests(unittest.TestCase):
    def test_real_shards_cover_queue(self):
        with tempfile.TemporaryDirectory() as d:
            old = aggregate.OUT
            aggregate.OUT = Path(d) / "out"
            try:
                result = aggregate.run()
                self.assertEqual(sum(result["labels"].values()), 119)
                self.assertEqual(result["labels"]["keep"], 27)
            finally:
                aggregate.OUT = old


if __name__ == "__main__":
    unittest.main()
