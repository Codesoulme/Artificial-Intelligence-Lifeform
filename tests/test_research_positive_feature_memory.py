import unittest
import torch
from spora.research_positive_feature_memory import features, read, write


class PositiveMemoryTests(unittest.TestCase):
    def test_aggregated_state_equals_pairwise_kernel(self):
        torch.manual_seed(19)
        k, v, q = torch.randn(6, 4), torch.randn(6, 3), torch.randn(5, 4)
        state = write(k, v)
        pair = features(q) @ features(k).T
        expected = (pair @ v) / pair.sum(-1, keepdim=True)
        torch.testing.assert_close(read(q, state), expected)
        a, b = write(k[:2], v[:2]), write(k[2:], v[2:])
        torch.testing.assert_close(read(q, (a[0] + b[0], a[1] + b[1])), expected)

    def test_rejects_empty_state(self):
        with self.assertRaises(ValueError):
            read(torch.ones(1, 4), (torch.zeros(4, 3), torch.zeros(4)))
