import tempfile
import unittest
from pathlib import Path

import torch
from types import SimpleNamespace

from spora.natural_write_heads import NaturalWriteHeadBundle, ValueSpanHead


class NaturalWriteHeadTests(unittest.TestCase):
    def test_value_head_shape(self):
        head = ValueSpanHead(8)
        accepted, boundary = head(torch.zeros(2, 2, 5, 8))
        self.assertEqual(accepted.shape, (2,))
        self.assertEqual(boundary.shape, (2, 5, 2))

    def test_bundle_freezes_heads_and_exposes_provenance(self):
        bundle = NaturalWriteHeadBundle(
            torch.nn.Linear(16, 6), ValueSpanHead(8), .1, -2.,
            artifact_sha256='a' * 64, base_sha256='b' * 64,
            checkpoint_sha256='c' * 64, feature_variant_sha256='d' * 64)
        self.assertFalse(any(parameter.requires_grad for parameter in bundle.parameters()))
        self.assertEqual(bundle.config_dict()['format'], 'spora-natural-write-coverage-heads-v1')


if __name__ == '__main__':
    unittest.main()
