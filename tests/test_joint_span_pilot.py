import unittest

import torch

from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text, token_span


class JointSpanPilotTests(unittest.TestCase):
    def test_span_offsets_and_unrepresentable_boundary(self):
        offsets = torch.tensor([[0, 0], [10, 12], [12, 13], [13, 14]])
        allowed = torch.tensor([False, True, True, True])
        self.assertEqual(token_span(offsets, allowed, 10, '小林月吗', '小林月'), 7)
        self.assertEqual(selected_text(7, offsets, 10, '小林月吗'), '小林月')
        self.assertIsNone(token_span(offsets, allowed, 10, '小林月吗', '林月'))
        self.assertEqual(token_span(offsets, allowed, 10, '小林月吗', None), 0)

    def test_invalid_spans_cannot_win(self):
        head = JointSpan(4, width=3)
        logits = head(torch.randn(4, 4), torch.tensor([False, True, True, False]))
        self.assertEqual(logits.shape, (17,))
        for left, right in ((0, 1), (2, 1), (1, 3)):
            self.assertLess(logits[1 + left * 4 + right].item(), -1e8)
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
