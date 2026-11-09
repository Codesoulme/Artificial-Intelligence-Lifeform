import unittest

import torch

from scripts.deepseek_name_span_probe import oracle_span_mask


class NameSpanTests(unittest.TestCase):
    def test_selects_only_provided_name_tokens(self):
        offsets = torch.tensor([[[0, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]])
        mask, crossings = oracle_span_mask(offsets, torch.ones(1, 6), ['u:小林的值？'], ['小林的值？'], ['小林'])
        self.assertEqual(mask.tolist(), [[False, True, True, False, False, False]])
        self.assertEqual(crossings, [False])

    def test_marks_crossing_and_rejects_ambiguous_name(self):
        offsets = torch.tensor([[[0, 2], [2, 5], [5, 7]]])
        mask, crossings = oracle_span_mask(offsets, torch.ones(1, 3), ['u:小林的值？'], ['小林的值？'], ['小林'])
        self.assertEqual(mask.tolist(), [[False, True, False]])
        self.assertEqual(crossings, [True])
        with self.assertRaises(ValueError):
            oracle_span_mask(torch.tensor([[[0, 5]]]), torch.ones(1, 1), ['小林和小林'], ['小林和小林'], ['小林'])


if __name__ == '__main__':
    unittest.main()
