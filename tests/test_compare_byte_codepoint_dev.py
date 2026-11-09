import math
import unittest

import torch
from torch import nn

from scripts.compare_byte_codepoint_dev import representable_spans, text_nll
from spora.codepoint import CodepointVocabulary


class UniformModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, state):
        return torch.zeros(*tokens.shape, 300, device=tokens.device) + self.anchor


class CompareByteCodepointTests(unittest.TestCase):
    def test_spans_exclude_unknowns_without_reordering(self):
        vocab = CodepointVocabulary(['甲', '乙'])
        self.assertEqual(representable_spans('丙甲乙丁甲丙', vocab), ['甲乙', '甲'])
        self.assertEqual(representable_spans('丙丁', vocab), [])

    def test_each_target_counted_once_across_windows(self):
        model = UniformModel()
        for length in (2, 3, 8):
            self.assertAlmostEqual(text_nll(model, None, [1, 2, 3, 4, 5], length),
                                   5 * math.log(300), places=5)
        with self.assertRaises(ValueError):
            text_nll(model, None, [], 4)


if __name__ == '__main__':
    unittest.main()
