import unittest
from spora.research_identity_threshold import calibrate,score


class IdentityThresholdTests(unittest.TestCase):
    def test_ties_rejected_at_threshold(self):
        negatives=[{'known':False,'scores':[x,0.]} for x in (4.,3.,3.,2.)]
        threshold=calibrate(negatives,1)
        self.assertEqual(threshold,3.)
        self.assertEqual(score(negatives,threshold)['false_accept_unknown'],1)

    def test_known_wrong_does_not_count_as_correct(self):
        rows=[{'known':True,'target':0,'scores':[1.,2.,0.]},
              {'known':True,'target':0,'scores':[1.,0.,0.]}]
        result=score(rows,0.)
        self.assertEqual(result['wrong_known'],1)
        self.assertEqual(result['correct_known'],1)
        self.assertEqual(score(rows,1.)['rejected_known'],1)
