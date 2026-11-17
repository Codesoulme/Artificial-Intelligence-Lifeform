import unittest
import torch
from spora.research_random_feature_attention import log_features, read, write


class RandomFeatureMemoryTests(unittest.TestCase):
    def test_aggregated_equals_explicit_kernel(self):
        torch.manual_seed(19)
        q, k, v, r = torch.randn(3, 4), torch.randn(5, 4), torch.randn(5, 2), torch.randn(30, 4)
        state = write(k, v, r)
        lq, lk = log_features(q, r), log_features(k, r)
        fq = (lq - lq.max(-1, keepdim=True).values).exp()
        fk = (lk - lk.max()).exp()
        weights = fq @ fk.T
        expected = weights @ v / weights.sum(-1, keepdim=True)
        torch.testing.assert_close(read(q, r, state), expected)

    def test_invalid_memory_rejected(self):
        with self.assertRaises(ValueError):
            read(torch.ones(1, 2), torch.ones(3, 2), (torch.zeros(3, 2), torch.zeros(3), torch.tensor(0.)))
