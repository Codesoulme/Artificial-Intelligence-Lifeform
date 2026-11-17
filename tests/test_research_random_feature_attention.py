import unittest
import io
import torch
from spora.research_random_feature_attention import log_features, merge, read, write


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

    def test_merge_rescales_and_survives_serialization(self):
        torch.manual_seed(81)
        keys, values, queries = torch.randn(19, 4), torch.randn(19, 3), torch.randn(7, 4)
        keys[5:9] *= 8
        projection = torch.randn(30, 4)
        whole = write(keys, values, projection)
        blocks = [write(keys[i:i+4], values[i:i+4], projection) for i in range(0, 19, 4)]
        merged = blocks[0]
        for block in blocks[1:]:
            merged = merge(merged, block)
        for a, b in zip(whole, merged):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        buffer = io.BytesIO()
        torch.save(merged, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        torch.testing.assert_close(read(queries, projection, restored), read(queries, projection, whole))

    def test_bad_state_rejected(self):
        with self.assertRaises(ValueError):
            merge((torch.zeros(3, 2), torch.zeros(3), torch.tensor(float('nan'))),
                  (torch.zeros(3, 2), torch.zeros(3), torch.tensor(0.)))
