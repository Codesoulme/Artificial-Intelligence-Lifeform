import unittest
from unittest.mock import patch

import torch

from scripts.deepseek_spora_joint_span_pilot import JointSpan, selected_text, token_span
from scripts.deepseek_spora_joint_span_crossed_pilot import build_records, TRAIN_TEMPLATES
from scripts.deepseek_spora_joint_span_diverse_pilot import build_records as build_diverse, UNSEEN_SUFFIXES


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

    def test_crossed_data_breaks_template_variant_pairing(self):
        training = [{'text': 'train'}]
        validation = [{'text': 'validation'}]
        train_names = {f'甲{i:02}' for i in range(48)}
        eval_names = [f'乙{i:02}' for i in range(20)]
        with patch('scripts.deepseek_spora_joint_span_crossed_pilot.extract_names',
                   side_effect=[train_names, set()]), patch(
                   'scripts.deepseek_spora_joint_span_crossed_pilot.select_names',
                   return_value=[('digest', name) for name in eval_names]):
            train, evaluation, count = build_records(training, validation)
        self.assertEqual(count, 48)
        questions = {r['question'] for r in train}
        for subject in ('甲00', '甲00子', '小甲00'):
            for template in TRAIN_TEMPLATES:
                self.assertIn(template.format(name=subject), questions)
        self.assertFalse(questions & {r['question'] for r in evaluation})
        self.assertEqual(sum(r['group'] == 'suffix_unseen_unseen' for r in evaluation), 20)

    def test_diverse_data_reserves_new_suffixes_and_dialogue(self):
        prior_eval = [{'group': 'known_unseen', 'reference': f'乙{i:02}'} for i in range(20)]
        with patch('scripts.deepseek_spora_joint_span_diverse_pilot.crossed_records',
                   return_value=([], prior_eval, 2)), patch(
                   'scripts.deepseek_spora_joint_span_diverse_pilot.extract_names',
                   side_effect=[{'甲00', '甲01'}, set()]):
            train, evaluation, _ = build_diverse([{'text': 'train'}], [{'text': 'validation'}])
        subjects = [row['subject'] for row in train if row['subject'] is not None]
        self.assertFalse(any(subject.endswith(UNSEEN_SUFFIXES) for subject in subjects))
        self.assertEqual(sum(row['group'] == 'suffix_lan_new_wording' for row in evaluation), 20)
        self.assertEqual(sum(row['group'] == 'multiple_new_wording' for row in evaluation), 8)


if __name__ == '__main__':
    unittest.main()
