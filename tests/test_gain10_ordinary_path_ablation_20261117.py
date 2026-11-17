import unittest
from scripts import deepseek_spora_continuous_attention_gain10_ordinary_20261117 as audit


class OrdinaryAblationSummaryTests(unittest.TestCase):
    def test_separates_all_memory_from_cross_branch_effect(self):
        rows = [{'variants': [{'token_ids': [1]},
                              {'token_ids': [2], 'cross_off_token_ids': [2]},
                              {'token_ids': [3], 'cross_off_token_ids': [4]}]}]
        old = audit.CROSS_ABLATION
        try:
            audit.CROSS_ABLATION = True
            self.assertEqual(audit.variant_summary(rows), {'forced_memory_conditions': 2,
                'changed_conditions': 2, 'cross_ablation_changed_conditions': 1})
        finally:
            audit.CROSS_ABLATION = old
