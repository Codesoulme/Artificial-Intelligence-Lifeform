import unittest

from scripts import deepseek_spora_diverse_kl_correction_audit as data
from scripts import deepseek_spora_internal_router_probe as probe
from scripts import deepseek_spora_multirelation_correction_audit as multi


class RouterProbeDesignTests(unittest.TestCase):
    def test_training_examples_are_unique_and_held_fact_questions_are_unseen(self):
        train, _ = data.load_reviewed_prompts()
        rows = probe.training_rows(train)
        self.assertEqual(len(rows), 78)
        questions = {row['question'] for row in rows}
        held = [case for case in multi.examples(multi.CORRECTED)
                if case['group'] in ('old_held', 'new_held', 'unknown_relation')]
        self.assertTrue(all(case['question'] not in questions for case in held))
        self.assertEqual(probe.expected_route('old_held'), 'known')
        self.assertEqual(probe.expected_route('unknown_fresh'), 'unknown')
        self.assertEqual(probe.expected_route('ordinary_held'), 'ordinary')


if __name__ == '__main__':
    unittest.main()
