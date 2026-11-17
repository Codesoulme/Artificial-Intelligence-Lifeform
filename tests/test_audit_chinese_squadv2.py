import unittest
from scripts.audit_chinese_squadv2_20261117 import audit


class ChineseSquadAuditTests(unittest.TestCase):
    def test_offsets_and_flag_mismatch(self):
        base = {'context': '小林喜欢茶。', 'question': '谁？'}
        rows = [{**base, 'id': 'a', 'is_impossible': False,
                 'answers': {'text': ['小林'], 'answer_start': [0]}},
                {**base, 'id': 'b', 'is_impossible': True,
                 'answers': {'text': [], 'answer_start': []}},
                {**base, 'id': 'c', 'is_impossible': True,
                 'answers': {'text': ['小林'], 'answer_start': [1]}}]
        result = audit(rows)
        self.assertEqual((result['answerable'], result['unanswerable']), (1, 2))
        self.assertEqual((result['flag_mismatch'], result['invalid_offset']), (1, 1))
