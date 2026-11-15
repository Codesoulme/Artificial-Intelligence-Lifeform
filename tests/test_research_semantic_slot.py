import unittest

import torch
from torch import nn

from spora.research_semantic_slot import (GainSemanticSlotBridge, ScaledSemanticSlotLMHead,
                                           SemanticSlotBridge, SequenceSemanticSlotLMHead,
                                           install, install_gain, install_scaled,
                                           install_sequence_scaled, use_slot, use_slot_sequence)


class ToyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(3, 4, bias=False)


class ResearchSemanticSlotTests(unittest.TestCase):
    def test_zero_bridge_preserves_head_without_slot(self):
        model = ToyCausalLM()
        bridge = install(model, rank=1)
        hidden = torch.tensor([[[1., 2., 3.], [2., 1., 0.]]])
        expected = model.lm_head.base(hidden)
        torch.testing.assert_close(model.lm_head(hidden), expected)
        self.assertIsInstance(bridge, SemanticSlotBridge)

    def test_slot_changes_only_scoped_call_and_has_gradient(self):
        model = ToyCausalLM()
        bridge = install(model, rank=1)
        hidden = torch.tensor([[[1., 2., 3.], [2., 1., 0.]]])
        slot = torch.tensor([0.5, -0.25, 0.75])
        ordinary = model.lm_head(hidden)
        with use_slot(slot):
            adapted = model.lm_head(hidden)
        self.assertTrue(torch.equal(ordinary, model.lm_head(hidden)))
        self.assertTrue(torch.equal(ordinary, adapted))
        loss = adapted.square().sum()
        loss.backward()
        self.assertIsNotNone(bridge.up.weight.grad)
        self.assertGreater(float(bridge.up.weight.grad.abs().sum()), 0.)

    def test_scaled_head_uses_slot_only_inside_scope(self):
        model = ToyCausalLM()
        install_scaled(model, scale=2.)
        self.assertIsInstance(model.lm_head, ScaledSemanticSlotLMHead)
        hidden = torch.tensor([[[1., 2., 3.], [2., 1., 0.]]])
        slot = torch.tensor([0.5, -0.25, 0.75])
        ordinary = model.lm_head(hidden)
        with use_slot(slot):
            adapted = model.lm_head(hidden)
        self.assertFalse(torch.equal(ordinary, adapted))
        self.assertTrue(torch.equal(ordinary, model.lm_head(hidden)))

    def test_gain_bridge_starts_with_identity_path(self):
        model = ToyCausalLM()
        bridge = install_gain(model, rank=1, initial_gain=3.)
        self.assertIsInstance(bridge, GainSemanticSlotBridge)
        slot = torch.tensor([0.5, -0.25, 0.75])
        with use_slot(slot):
            output = model.lm_head(torch.zeros(1, 2, 3))
        expected = model.lm_head.base((3. * slot).reshape(1, 1, 3).expand(1, 2, 3))
        torch.testing.assert_close(output, expected)

    def test_sequence_head_consumes_vectors_in_order(self):
        model = ToyCausalLM()
        install_sequence_scaled(model, scale=2.)
        self.assertIsInstance(model.lm_head, SequenceSemanticSlotLMHead)
        hidden = torch.zeros(1, 2, 3)
        first = torch.tensor([1., 0., 0.])
        second = torch.tensor([0., 1., 0.])
        with use_slot_sequence([first, second]):
            first_output = model.lm_head(hidden)
            second_output = model.lm_head(hidden)
            exhausted_output = model.lm_head(hidden)
        torch.testing.assert_close(first_output, model.lm_head.base((2 * first).reshape(1, 1, 3).expand_as(hidden)))
        torch.testing.assert_close(second_output, model.lm_head.base((2 * second).reshape(1, 1, 3).expand_as(hidden)))
        torch.testing.assert_close(exhausted_output, model.lm_head.base(hidden))


if __name__ == '__main__':
    unittest.main()
