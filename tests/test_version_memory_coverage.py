import unittest

import torch

from spora.research_versioned_memory_pointer import VersionCoverageBias, VersionReplyActions


class VersionMemoryCoverageTests(unittest.TestCase):
    def test_ignores_prompt_and_adds_only_a_soft_continuation_bias(self):
        bias = VersionCoverageBias([2, 3, 4])
        scores = torch.zeros(1, 6)
        scores[0, 5] = 10.
        self.assertIs(bias(torch.tensor([[0, 2]]), scores), scores)
        shifted = bias(torch.tensor([[0, 2, 2]]), scores)
        self.assertEqual(shifted.argmax(-1).item(), 5)
        self.assertEqual(shifted[0, 3].item(), 8.)
        self.assertEqual(scores[0, 3].item(), 0.)

    def test_follows_an_overlapping_prefix_without_repeating_a_completed_value(self):
        bias = VersionCoverageBias([2, 2, 3])
        self.assertEqual(bias.next_token((2, 2)), 3)
        self.assertEqual(bias.next_token((2, 0, 2)), 2)
        self.assertIsNone(bias.next_token((2, 2, 3, 0, 2)))
        self.assertIsNone(bias.next_token((0, 1)))

    def test_empty_and_single_token_values_do_not_require_continuation(self):
        scores = torch.zeros(1, 6)
        for value in ([], [2]):
            with self.subTest(value=value):
                bias = VersionCoverageBias(value)
                self.assertIsNone(bias.next_token((2,)))
                self.assertIs(bias(torch.tensor([[0, 2]]), scores), scores)
                self.assertIsNone(bias.prompt_length)


class VersionReplyActionTests(unittest.TestCase):
    def test_whole_value_action_competes_once_and_respects_the_remaining_budget(self):
        head = VersionReplyActions(3, [4])
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([9., 0.]))
        hidden = torch.zeros(3)
        logits = torch.zeros(6)
        self.assertIsNone(head.choose(hidden, logits, copied=False, room_for_copy=True, memory_fraction=.95))
        self.assertEqual(head.choose(hidden, logits, copied=True, room_for_copy=True, memory_fraction=.95), 4)
        self.assertEqual(head.choose(hidden, logits, copied=False, room_for_copy=False, memory_fraction=.95), 4)

    def test_an_ordinary_token_can_win_before_a_copy_occurs(self):
        head = VersionReplyActions(3, [4])
        with torch.no_grad():
            head.actions.weight.zero_()
            head.actions.bias.copy_(torch.tensor([-9., 0.]))
        logits = torch.tensor([15., 0., 0., 0., 0., 0.])
        self.assertEqual(head.choose(torch.zeros(3), logits, copied=False, room_for_copy=True, memory_fraction=.2), 0)
