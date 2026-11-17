import unittest
from scripts.compare_squadv2_independent_review_20261117 import compare, INDEPENDENT
from scripts.review_squadv2_noanswer_20261117 import REVIEW


class IndependentReviewTests(unittest.TestCase):
    def test_conservative_intersection_and_manifest(self):
        selection = {'rows': [{'digest': k, 'kind': 'unknown'} for k in INDEPENDENT]}
        prior = {'rows': [{'digest': k, 'group': 'unknown',
                           'heads': {n: {'selected': True} for n in ('prior', 'natural', 'question')}}
                          for k in INDEPENDENT]}
        result = compare(selection, prior)
        self.assertEqual(result['agreement'], 15)
        self.assertEqual(result['conservative']['supported']['total'], 9)
        self.assertEqual(set(REVIEW), set(INDEPENDENT))
        with self.assertRaises(ValueError):
            compare({'rows': []}, prior)
