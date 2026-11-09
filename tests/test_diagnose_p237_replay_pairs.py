import unittest

from scripts.diagnose_p237_replay_pairs import aggregate


class PairAggregateTests(unittest.TestCase):
    def test_weighted_and_exposure_groups(self):
        rows = [{'targets': 2, 'presentations': 1, 'parent': {'context': 4., 'last_user': 6.},
                 'replay': {'context': 2., 'last_user': 4.}},
                {'targets': 1, 'presentations': 0, 'parent': {'context': 2., 'last_user': 3.},
                 'replay': {'context': 1., 'last_user': 2.}}]
        result = aggregate(rows)
        self.assertEqual(result['all']['parent_context_nll'], 2.)
        self.assertEqual(result['seen']['replay_context_nll'], 1.)
        self.assertEqual(result['unseen']['parent_last_user_nll'], 3.)


if __name__ == '__main__':
    unittest.main()
