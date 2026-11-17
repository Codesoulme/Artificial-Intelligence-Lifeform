import unittest
from scripts.deepseek_dulemon_evidence_spora_long_context_20261117 import check_position_budget


class IntegratedPositionBudgetTests(unittest.TestCase):
    def test_both_prompt_and_generation_must_fit_each_layer(self):
        check_position_budget(224, 32, {6: 768, 12: 256})
        with self.assertRaises(ValueError):
            check_position_budget(225, 32, {6: 768, 12: 256})
        for invalid in (0, -1, True):
            with self.assertRaises(ValueError):
                check_position_budget(invalid, 32, {6: 768})
        with self.assertRaises(ValueError):
            check_position_budget(100, 32, {})
