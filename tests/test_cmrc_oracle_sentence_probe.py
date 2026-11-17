import unittest
from scripts.deepseek_cmrc_oracle_sentence_probe_20261117 import oracle_sentence


class OracleSentenceTests(unittest.TestCase):
    def test_selects_answer_sentence_and_adjusts_offsets(self):
        row = {'text': '第一句。小林喜欢茶。还有其他。\n问题：谁喜欢茶？', 'left': 4, 'right': 6}
        result = oracle_sentence(row)
        self.assertEqual(result['text'], '小林喜欢茶。\n问题：谁喜欢茶？')
        self.assertEqual(result['text'][result['left']:result['right']], '小林')

    def test_rejects_answer_crossing_sentence_boundary(self):
        row = {'text': '甲。乙。\n问题：谁？', 'left': 0, 'right': 3}
        with self.assertRaises(ValueError):
            oracle_sentence(row)
