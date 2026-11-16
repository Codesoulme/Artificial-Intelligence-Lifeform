import unittest

from scripts.deepseek_dulemon_context_intent_challenge_20261117 import summary


class ContextIntentChallengeTests(unittest.TestCase):
    def test_counts_true_and_false_accepts_for_each_input(self):
        rows = [
            {'read': True, 'single': False, 'context': True},
            {'read': False, 'single': True, 'context': True},
            {'read': False, 'single': False, 'context': False},
        ]
        self.assertEqual(summary(rows, 'single'),
                         {'total': 3, 'positives': 1, 'true_accepts': 0, 'false_accepts': 1})
        self.assertEqual(summary(rows, 'context'),
                         {'total': 3, 'positives': 1, 'true_accepts': 1, 'false_accepts': 1})
