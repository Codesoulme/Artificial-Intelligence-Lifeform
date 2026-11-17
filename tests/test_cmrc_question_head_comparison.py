import unittest
import torch
from scripts.deepseek_cmrc_question_head_comparison_20261117 import question_start, question_tokens


class QuestionMaskTests(unittest.TestCase):
    def test_explicit_boundaries(self):
        self.assertEqual(question_start('证据。谁？'), 3)
        self.assertEqual(question_start('证据。\n问题：谁？'), 7)
        with self.assertRaises(ValueError):
            question_start('没有边界')

    def test_mask_excludes_evidence_and_padding(self):
        item = {'start': 10, 'row': {'text': '甲。谁？'},
                'offsets': torch.tensor([[10, 11], [11, 12], [12, 13], [13, 14], [0, 0]]),
                'subject_mask': torch.tensor([True, True, True, True, False])}
        self.assertEqual(question_tokens(item).tolist(), [False, False, True, True, False])
