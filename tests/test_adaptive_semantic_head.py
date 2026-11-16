import unittest

import torch
from torch import nn
from spora.research_adaptive_semantic_head import AdaptiveSemanticHead, use_adaptive_slot


class AdaptiveSemanticHeadTests(unittest.TestCase):
    def test_inactive_path_and_context_cleanup(self):
        base = nn.Linear(4, 6, bias=False)
        head = AdaptiveSemanticHead(base, rank=2)
        hidden = torch.randn(2, 3, 4)
        expected = base(hidden)
        torch.testing.assert_close(head(hidden), expected)
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with use_adaptive_slot(torch.randn(4)):
                self.assertFalse(torch.equal(head(hidden), expected))
                raise RuntimeError('failure')
        torch.testing.assert_close(head(hidden), expected)

    def test_slot_persists_across_calls_and_rejects_nesting(self):
        base = nn.Linear(4, 6, bias=False)
        head = AdaptiveSemanticHead(base, rank=2)
        hidden = torch.randn(1, 1, 4)
        with use_adaptive_slot(torch.randn(4)):
            torch.testing.assert_close(head(hidden), head(hidden))
            with self.assertRaisesRegex(ValueError, 'Nested'):
                with use_adaptive_slot(torch.randn(4)):
                    pass

    def test_trainable_parameters_receive_gradient(self):
        head = AdaptiveSemanticHead(nn.Linear(4, 6, bias=False), rank=2)
        hidden = torch.randn(3, 2, 4)
        output = head.adjusted(hidden, torch.randn(3, 4))
        output.square().sum().backward()
        self.assertIsNotNone(head.stop.grad)
        self.assertIsNotNone(head.gates.weight.grad)
        self.assertIsNone(head.base.weight.grad)


if __name__ == '__main__':
    unittest.main()
