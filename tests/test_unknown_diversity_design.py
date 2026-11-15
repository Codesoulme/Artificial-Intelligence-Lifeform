import collections
import unittest

from scripts import deepseek_spora_multirelation_correction_audit as prior
from scripts import deepseek_spora_unknown_diversity_audit as audit


class UnknownDiversityDesignTests(unittest.TestCase):
    def test_design_and_matched_training_schedule(self):
        audit.validate_design()
        self.assertEqual(audit.STEPS, 84)
        original = [audit.unknown_pair(step, False) for step in range(audit.STEPS)]
        diverse = [audit.unknown_pair(step, True) for step in range(audit.STEPS)]
        self.assertEqual(sum(name in audit.EXTRA_TRAIN for name, _ in original), 0)
        self.assertEqual(sum(name in audit.EXTRA_TRAIN for name, _ in diverse), 42)
        self.assertEqual(len(set(diverse) & set(audit.EXTRA_PAIRS)), len(audit.EXTRA_PAIRS))
        self.assertFalse(set(diverse) & set(prior.CORRECTED))
        fact_schedule = collections.Counter(((step // 2) % len(prior.CORRECTED), step % 2)
                                            for step in range(audit.STEPS))
        self.assertEqual(len(fact_schedule), 14)
        self.assertEqual(set(fact_schedule.values()), {6})


if __name__ == '__main__':
    unittest.main()
