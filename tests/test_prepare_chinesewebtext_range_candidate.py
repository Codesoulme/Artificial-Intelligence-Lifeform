import tempfile
import unittest
from pathlib import Path

from scripts.prepare_chinesewebtext_range_candidate import cjk_ratio, filter_reasons, stable_rank


class ChineseWebTextCandidateTests(unittest.TestCase):
    def test_cjk_ratio_and_mechanical_gate(self):
        text = '中文内容' * 80
        row = {
            'text': text,
            'quality_score': 0.95,
            'toxicity': {'label': 0, 'score': 0.01},
            'domain': {'single_label': 'general'},
        }
        self.assertGreater(cjk_ratio(text), 0.9)
        self.assertEqual(filter_reasons(row), [])

    def test_rejects_low_quality_and_pii(self):
        row = {
            'text': ('普通文本' * 80) + ' test@example.com',
            'quality_score': 0.4,
            'toxicity': {'label': 0, 'score': 0.01},
            'domain': {'single_label': 'general'},
        }
        reasons = filter_reasons(row)
        self.assertIn('quality_below_0.9', reasons)
        self.assertIn('email_pattern', reasons)

    def test_rank_is_stable(self):
        self.assertEqual(stable_rank(1, '文本'), stable_rank(1, '文本'))
        self.assertNotEqual(stable_rank(1, '文本'), stable_rank(2, '文本'))


if __name__ == '__main__':
    unittest.main()
