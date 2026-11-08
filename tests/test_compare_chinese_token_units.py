import unittest

from scripts.compare_chinese_token_units import summarize


class TokenUnitComparisonTests(unittest.TestCase):
    def test_reads_only_train_and_development_and_has_expected_order(self):
        result = summarize()
        self.assertFalse(result["test_split_read"])
        units = result["units"]
        self.assertGreater(units["byte"]["content_units"], units["codepoint"]["content_units"])
        self.assertGreater(units["codepoint"]["content_units"], units["bpe"]["content_units"])
        self.assertEqual(result["splits_read"], {"train": 306, "development": 8})


if __name__ == "__main__":
    unittest.main()
