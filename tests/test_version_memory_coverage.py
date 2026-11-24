import unittest

import torch

from spora.research_versioned_memory_pointer import VersionCoverageBias


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
