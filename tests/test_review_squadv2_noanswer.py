import unittest
from scripts.review_squadv2_noanswer_20261117 import summarize


class NoAnswerReviewTests(unittest.TestCase):
    def test_groups_and_predictions_are_joined_by_digest(self):
        selected = {'rows': [{'digest': 'a', 'kind': 'unknown'}, {'digest': 'b', 'kind': 'unknown'}]}
        prior = {'rows': [
            {'digest': 'b', 'group': 'unknown', 'heads': {name: {'selected': True} for name in ('prior', 'natural', 'question')}},
            {'digest': 'a', 'group': 'unknown', 'heads': {name: {'selected': False} for name in ('prior', 'natural', 'question')}}]}
        result = summarize(selected, prior, {'a': ('supported', 'x'), 'b': ('ambiguous', 'y')})
        self.assertEqual(result['supported']['natural'], 0)
        self.assertEqual(result['ambiguous']['question'], 1)
        with self.assertRaises(ValueError):
            summarize(selected, prior, {'a': ('supported', 'x')})
