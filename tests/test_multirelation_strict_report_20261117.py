import unittest
from scripts.deepseek_multirelation_strict_report_20261117 import score_phase


class MultiRelationStrictReportTests(unittest.TestCase):
    def test_pair_requires_correct_subject_relation_and_both_values(self):
        base = {'group': 'held_name', 'name': '小赵', 'question': '问题', 'expected': '果汁'}
        rows = [{**base, 'condition': 'memory', 'stored_expected': '果汁', 'answer': '小赵喜欢喝果汁。'},
                {**base, 'condition': 'swapped', 'stored_expected': '可乐', 'answer': '小林喜欢喝可乐。'}]
        result = score_phase(rows)
        self.assertEqual(result['pair_total'], 1)
        self.assertEqual(result['pair_both_strict'], 0)
        rows[1]['answer'] = '小赵喜欢喝可乐。'
        self.assertEqual(score_phase(rows)['pair_both_strict'], 1)

    def test_duplicate_condition_is_rejected(self):
        row = {'group': 'held_name', 'name': '小赵', 'question': '问题', 'expected': '果汁',
               'condition': 'memory', 'stored_expected': '果汁', 'answer': '小赵喜欢喝果汁。'}
        with self.assertRaises(ValueError):
            score_phase([row, row])
