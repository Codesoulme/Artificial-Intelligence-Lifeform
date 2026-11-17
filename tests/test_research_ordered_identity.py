import unittest
import torch
from spora.research_ordered_identity import ordered_similarity,select_ordered


class OrderedIdentityTests(unittest.TestCase):
    def test_length_order_and_exact(self):
        a=torch.tensor([[1.,0.],[0.,1.]])
        self.assertAlmostEqual(ordered_similarity(a,a),1.)
        self.assertAlmostEqual(ordered_similarity(a,a.flip(0)),0.)
        self.assertEqual(ordered_similarity(a,torch.cat((a,a[:1]))),-1.)

    def test_zero_token_rejected(self):
        with self.assertRaises(ValueError):ordered_similarity(torch.zeros(1,2),torch.ones(1,2))

    def test_ties_rejected_and_unknown_length(self):
        v=torch.tensor([[1.,0.],[0.,1.]])
        result=select_ordered(v,[v,v.clone()],0.2)
        self.assertIsNone(result['index']);self.assertTrue(result['ambiguous'])
        self.assertIsNone(select_ordered(torch.cat((v,v[:1])),[v],0.2)['index'])
