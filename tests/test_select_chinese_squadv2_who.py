import unittest
from scripts.select_chinese_squadv2_who_20261117 import select


class ChineseSquadSelectionTests(unittest.TestCase):
    def test_answerable_and_unanswerable_and_distinct_context(self):
        rows = [
            {'id': 'a', 'context': '小林喜欢茶。', 'question': '谁喜欢茶？', 'is_impossible': False,
             'answers': {'text': ['小林'], 'answer_start': [0]}},
            {'id': 'b', 'context': '小王喜欢咖啡。', 'question': '谁喜欢茶？', 'is_impossible': True,
             'answers': {'text': [], 'answer_start': []}},
            {'id': 'c', 'context': '小林喜欢茶。', 'question': '谁喜欢咖啡？', 'is_impossible': True,
             'answers': {'text': [], 'answer_start': []}},
        ]
        chosen, eligible = select(rows, per_group=1)
        self.assertEqual(eligible, {'known': 1, 'unknown': 2})
        self.assertEqual([r['kind'] for r in chosen], ['known', 'unknown'])
        self.assertNotEqual(chosen[0]['context_digest'], chosen[1]['context_digest'])

    def test_rejects_inconsistent_unanswerable_flag(self):
        rows = [{'id': 'x', 'context': '小林喜欢茶。', 'question': '谁？', 'is_impossible': True,
                 'answers': {'text': ['小林'], 'answer_start': [0]}}]
        with self.assertRaises(ValueError):
            select(rows, per_group=1)
