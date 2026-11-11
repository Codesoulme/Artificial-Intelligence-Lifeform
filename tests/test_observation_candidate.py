import unittest

from spora.observation_candidate import ObservationCandidate, write_guard


class ObservationCandidateTests(unittest.TestCase):
    def test_unknown_text_value_cannot_enter_finite_memory(self):
        candidate = ObservationCandidate(3, '汽水', None, 'affirmative', 'direct', 1.0, 4, 0.99, '小林喜欢汽水')
        decision = write_guard(candidate, key_count=8, value_count=8, current_cursor=4)
        self.assertFalse(decision['write'])
        self.assertIn('finite_value_unresolved', decision['reasons'])

    def test_nonfactual_claim_is_rejected_even_with_a_known_value(self):
        candidate = ObservationCandidate(3, '水', 6, 'negation', 'direct', 1.0, 4, 0.99, '小林不喜欢水')
        decision = write_guard(candidate, key_count=8, value_count=8, current_cursor=4)
        self.assertFalse(decision['write'])
        self.assertIn('claim_is_not_affirmative', decision['reasons'])

    def test_complete_direct_candidate_is_only_a_decision(self):
        candidate = ObservationCandidate(3, '水', 6, 'affirmative', 'direct', 1.0, 4, 0.99, '小林喜欢水')
        self.assertEqual(write_guard(candidate, key_count=8, value_count=8, current_cursor=4),
                         {'write': True, 'reasons': []})

    def test_future_time_is_rejected(self):
        candidate = ObservationCandidate(3, '水', 6, 'affirmative', 'direct', 1.0, 5, 0.99, '小林喜欢水')
        decision = write_guard(candidate, key_count=8, value_count=8, current_cursor=4)
        self.assertFalse(decision['write'])
        self.assertIn('effective_time_unresolved', decision['reasons'])


if __name__ == '__main__':
    unittest.main()
