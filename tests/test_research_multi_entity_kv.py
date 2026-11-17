import unittest
import torch
from spora.research_multi_entity_kv import concatenate_kv


class MultiEntityKVTests(unittest.TestCase):
    def test_order_and_pair_alignment(self):
        a=(torch.tensor([[1.]]),torch.tensor([[2.]]))
        b=(torch.tensor([[3.]]),torch.tensor([[4.]]))
        k,v=concatenate_kv(a,b,reverse=True)
        torch.testing.assert_close(k,torch.tensor([[3.],[1.]]))
        torch.testing.assert_close(v,torch.tensor([[4.],[2.]]))

    def test_misaligned_lengths_rejected(self):
        with self.assertRaises(ValueError):
            concatenate_kv((torch.ones(2,1),torch.ones(1,1)),(torch.ones(1,1),torch.ones(1,1)))
