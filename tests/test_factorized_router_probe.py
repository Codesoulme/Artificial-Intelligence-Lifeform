import unittest

from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_factorized_router_probe as probe
from scripts import deepseek_spora_router_paraphrase_audit as prior


class FactorizedRouterProbeTests(unittest.TestCase):
    def test_oracle_span_requires_unique_name_and_covers_characters(self):
        self.assertEqual(probe.span_indices('小林喝什么', '小林', '问：小林喝什么',
                                            [(0, 2), (2, 4), (4, 7)]), [1])
        with self.assertRaises(ValueError):
            probe.span_indices('小林和小林', '小林', '小林和小林', [(0, 5)])
        with self.assertRaises(ValueError):
            probe.span_indices('小林喝什么', '小林', '小林喝什么', [(2, 5)])

    def test_training_questions_align_with_previous_augmentation(self):
        ordinary, _ = data.load_reviewed_prompts()
        rows = probe.training_rows(ordinary)
        previous, fresh = prior.design(ordinary)
        self.assertEqual(len(rows), 183)
        self.assertEqual({row['question'] for row in rows}, {row['question'] for row in previous})
        self.assertFalse({row['question'] for row in rows} & {row['question'] for row in fresh})


if __name__ == '__main__':
    unittest.main()
