import tempfile
import unittest
from pathlib import Path

from scripts.clean_open_paper_sample import cjk_ratio, emails


class OpenPaperSampleTests(unittest.TestCase):
    def test_signals(self):
        self.assertGreater(cjk_ratio('中文内容' * 100), 0.9)
        self.assertEqual(len(emails('作者 a@example.com')), 1)


if __name__ == '__main__':
    unittest.main()
