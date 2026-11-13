import unittest

import torch

from scripts.deepseek_spora_original_multi_span_pilot import spans_from_mask


class OriginalMultiSpanTests(unittest.TestCase):
    def setUp(self):
        self.question = '小林和阿宁？'
        self.text = '问:' + self.question
        self.offsets = torch.tensor([[2, 4], [4, 5], [5, 7], [7, 8]])

    def test_separate_selected_runs_form_two_spans(self):
        mask = torch.tensor([True, False, True, False])
        self.assertEqual(spans_from_mask(mask, self.offsets, self.text, self.question), ['小林', '阿宁'])

    def test_consecutive_tokens_form_one_span(self):
        mask = torch.tensor([True, True, True, False])
        self.assertEqual(spans_from_mask(mask, self.offsets, self.text, self.question), ['小林和阿宁'])

    def test_rejects_token_outside_question(self):
        offsets = self.offsets.clone()
        offsets[0] = torch.tensor([0, 2])
        with self.assertRaises(ValueError):
            spans_from_mask(torch.tensor([True, False, False, False]), offsets, self.text, self.question)

    def test_rejects_invalid_mask(self):
        with self.assertRaises(ValueError):
            spans_from_mask(torch.tensor([1, 0, 0, 0]), self.offsets, self.text, self.question)


if __name__ == '__main__':
    unittest.main()
