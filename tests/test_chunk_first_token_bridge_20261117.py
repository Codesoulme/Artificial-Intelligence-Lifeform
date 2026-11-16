import unittest

import torch
from torch.nn import functional as F

from scripts.deepseek_chunk_first_token_bridge_20261117 import FirstTokenBridge, extend_train_types, split_token_types


class FirstTokenBridgeTests(unittest.TestCase):
    def test_token_types_are_disjoint_and_deterministic(self):
        counts = {i: i+1 for i in range(30)}
        first = split_token_types(counts, 12, 6, 21)
        self.assertEqual(first, split_token_types(counts, 12, 6, 21))
        self.assertTrue(set(first[0]).isdisjoint(first[1]))
        self.assertEqual((len(first[0]), len(first[1])), (12, 6))
        with self.assertRaises(ValueError):
            split_token_types(counts, 29, 6, 21)

    def test_bridge_receives_gradient_through_frozen_output_head(self):
        torch.manual_seed(7)
        bridge = FirstTokenBridge(9, 12, hidden=8)
        head = torch.nn.Linear(12, 16, bias=False)
        head.requires_grad_(False)
        opt = torch.optim.AdamW(bridge.parameters(), lr=.01)
        inputs = torch.randn(4, 9)
        original = bridge.layers[-1].weight.detach().clone()
        logits = head(bridge(inputs))
        F.cross_entropy(logits, torch.tensor([1, 2, 3, 4])).backward()
        self.assertIsNotNone(bridge.layers[-1].weight.grad)
        self.assertGreater(float(bridge.layers[-1].weight.grad.norm()), 0)
        self.assertIsNone(head.weight.grad)
        opt.step()
        self.assertFalse(torch.equal(original, bridge.layers[-1].weight))

    def test_expansion_preserves_training_prefix_and_held_types(self):
        counts = {i: i+1 for i in range(30)}
        train, held = split_token_types(counts, 8, 6, 21)
        expanded = extend_train_types(counts, train, held, 20, 21)
        self.assertEqual(expanded[:len(train)], train)
        self.assertTrue(set(expanded).isdisjoint(held))
        self.assertEqual(len(set(expanded)), 20)
        self.assertEqual(expanded, extend_train_types(counts, train, held, 20, 21))
        with self.assertRaises(ValueError):
            extend_train_types(counts, train, held, 25, 21)
