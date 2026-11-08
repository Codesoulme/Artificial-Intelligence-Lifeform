import json
from pathlib import Path
import tempfile
import unittest

from scripts import p122_build_story_rereview_queue as queue


class RereviewQueueTests(unittest.TestCase):
    def test_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            old = queue.OUT
            queue.OUT = Path(d) / "existing"
            queue.OUT.mkdir()
            try:
                with self.assertRaises(FileExistsError):
                    queue.run()
            finally:
                queue.OUT = old

    def test_real_queue_has_119_flagged_records(self):
        with tempfile.TemporaryDirectory() as d:
            old = queue.OUT
            queue.OUT = Path(d) / "queue"
            try:
                result = queue.run()
                self.assertEqual(result["records"], 119)
                self.assertEqual(sum(result["first_review_labels"].values()), 119)
            finally:
                queue.OUT = old


if __name__ == "__main__":
    unittest.main()
