import io
import unittest
import torch
from spora.research_bound_feature_memory import BoundFeatureMemory


class BoundFeatureTests(unittest.TestCase):
    def test_restore_and_mismatches(self):
        torch.manual_seed(13)
        p = torch.randn(16, 4)
        x, v, q = torch.randn(8, 4), torch.randn(8, 3), torch.randn(2, 4)
        a = BoundFeatureMemory(p, base_sha256='a' * 64, layer=5, kv_head=0)
        state = a.merge(a.write(x[:3], v[:3]), a.write(x[3:], v[3:]))
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=True)
        torch.testing.assert_close(a.read(q, restored), a.read(q, a.write(x, v)))
        for b in (BoundFeatureMemory(p + 1, base_sha256='a'*64, layer=5, kv_head=0),
                  BoundFeatureMemory(p, base_sha256='b'*64, layer=5, kv_head=0),
                  BoundFeatureMemory(p, base_sha256='a'*64, layer=6, kv_head=0),
                  BoundFeatureMemory(p, base_sha256='a'*64, layer=5, kv_head=1)):
            with self.assertRaises(ValueError):
                b.read(q, restored)
            with self.assertRaises(ValueError):
                b.merge(restored, b.write(x, v))

    def test_external_projection_mutation_is_isolated(self):
        p = torch.randn(8, 4)
        memory = BoundFeatureMemory(p, base_sha256='a'*64, layer=0, kv_head=0)
        state = memory.write(torch.randn(2, 4), torch.randn(2, 3))
        p.zero_()
        self.assertTrue(torch.isfinite(memory.read(torch.randn(1, 4), state)).all())

    def test_nonfinite_memory_is_rejected(self):
        memory = BoundFeatureMemory(torch.randn(8, 4), base_sha256='a'*64, layer=0, kv_head=0)
        state = memory.write(torch.randn(2, 4), torch.randn(2, 3))
        state['payload'][0][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            memory.read(torch.randn(1, 4), state)
