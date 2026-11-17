import unittest
import torch
from spora.research_vector_identity_select import select_identity


class VectorIdentitySelectTests(unittest.TestCase):
    def test_correct_unknown_and_tie(self):
        records=torch.tensor([[1.,0.],[0.,1.]])
        self.assertEqual(select_identity(torch.tensor([1.,0.]),records,0.7)['index'],0)
        self.assertIsNone(select_identity(torch.tensor([-1.,0.]),records,0.7)['index'])
        tied=select_identity(torch.tensor([1.,1.]),records,0.6)
        self.assertIsNone(tied['index']);self.assertTrue(tied['ambiguous'])

    def test_zero_vectors_rejected(self):
        with self.assertRaises(ValueError):select_identity(torch.zeros(2),torch.ones(2,2),0.5)
