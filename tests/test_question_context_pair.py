import unittest
import torch
from spora.research_question_context_pair import question_rows, pair_features


class PairTests(unittest.TestCase):
    def test_question_alignment_is_not_row_position_assumption(self):
        rows = [dict(case=2, group='b', current_message_id='bq', variant='wrong_history'),
                dict(case=1, group='a', current_message_id='aq', variant='current_only'),
                dict(case=2, group='b', current_message_id='bq', variant='current_only')]
        self.assertEqual(question_rows(rows), [2, 1, 2])
        rows[0]['current_message_id'] = 'aq'
        with self.assertRaises(ValueError):
            question_rows(rows)

    def test_linear_control_is_linear_and_dimensions_match(self):
        c = torch.tensor([[1., 2.]])
        q = torch.tensor([[3., -4.]])
        base = pair_features(c, q, 'linear_pair')
        self.assertTrue(torch.allclose(pair_features(2*c, 2*q, 'linear_pair'), 2*base))
        self.assertEqual(base.shape, pair_features(c, q, 'interaction_pair').shape)
        self.assertFalse(torch.allclose(pair_features(2*c, 2*q, 'interaction_pair'),
                                       2*pair_features(c, q, 'interaction_pair')))

    def test_identical_input_difference_is_zero(self):
        x = torch.tensor([[.6, .8]])
        p = pair_features(x, x, 'interaction_pair')
        self.assertTrue(torch.equal(p[:, -2:], torch.zeros(1, 2)))
        with self.assertRaises(ValueError):
            pair_features(x, x, 'other')
