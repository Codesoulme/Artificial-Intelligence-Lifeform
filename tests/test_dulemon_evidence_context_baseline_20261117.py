import unittest
from scripts.deepseek_dulemon_evidence_context_baseline_20261117 import classify


class ContextBaselineTests(unittest.TestCase):
    def test_literal_indicators_do_not_credit_hallucination_as_unknown(self):
        result = classify('不知道', '篮球', '篮球')
        self.assertEqual(result, {'before_contains_unknown': True, 'before_leaks_expected': False,
                                  'after_contains_expected': True, 'after_contains_unknown': False})
        self.assertTrue(classify('篮球', '不知道', '篮球')['before_leaks_expected'])
        with self.assertRaises(ValueError):
            classify('不知道', '篮球', '')
