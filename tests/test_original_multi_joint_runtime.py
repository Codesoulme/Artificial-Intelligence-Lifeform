import unittest

import torch

from scripts.deepseek_spora_original_layer_joint_runtime_audit import spans_agree


class OriginalMultiJointRuntimeTests(unittest.TestCase):
    def test_equal_single_spans_pass(self):
        span = torch.tensor([False, True, True, False])
        self.assertTrue(spans_agree(span, span.clone(), True, True))

    def test_mismatch_or_multiple_spans_rejected(self):
        span = torch.tensor([False, True, False, False])
        self.assertFalse(spans_agree(span, torch.tensor([False, False, True, False]), True, True))
        self.assertFalse(spans_agree(span, span, True, False))
        self.assertFalse(spans_agree(span, span, False, True))

    def test_invalid_masks_rejected(self):
        span = torch.tensor([False, True])
        self.assertFalse(spans_agree(span, None, True, True))
        self.assertFalse(spans_agree(span, torch.tensor([0, 1]), True, True))
        self.assertFalse(spans_agree(span, torch.tensor([True]), True, True))
        self.assertFalse(spans_agree(span[None], span[None], True, True))


if __name__ == '__main__':
    unittest.main()
