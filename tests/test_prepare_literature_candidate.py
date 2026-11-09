import unittest

from scripts.prepare_literature_candidate import cjk_ratio


class LiteratureCandidateTests(unittest.TestCase):
    def test_cjk_ratio(self):
        self.assertGreater(cjk_ratio('红楼梦中的中文故事'), 0.9)
        self.assertLess(cjk_ratio('English words only'), 0.1)


if __name__ == '__main__':
    unittest.main()
