import unittest

import torch

from spora.research_name_role import (
    JointNameRoleHead, NameRoleHead, NameSpanHead, compare_name_scores, mention_features, select_name,
    select_nonoverlapping_spans, validate_enrollment,
)


class NameRoleTests(unittest.TestCase):
    def test_enrollment_does_not_fall_back_to_another_person_or_a_shorter_name(self):
        enrolled = [(0, 3, 1), (7, 10, 2)]
        self.assertIsNone(validate_enrollment((0, 4), enrolled))
        self.assertIsNone(validate_enrollment((3, 6), enrolled))
        self.assertEqual(validate_enrollment((7, 10), enrolled), (7, 10, 2))

    def test_span_selection_excludes_padding_and_prefers_higher_scoring_overlap(self):
        scores = torch.zeros(8, 3)
        scores[0] = torch.tensor([1., 2., 3.])
        scores[5] = torch.tensor([1., 10., 20.])
        scores[7, 0] = 30.
        self.assertEqual(select_nonoverlapping_spans(scores, 7, threshold=0.), [(0, 4), (5, 7)])
        self.assertEqual(NameSpanHead(5)(torch.rand(2, 7, 5)).shape, (2, 7, 3))

    def test_joint_score_requires_complete_name_and_superiority_to_abstention(self):
        decision = torch.tensor([[2., 1., 0.], [2., 1., 0.], [1., 0., 2.], [2., 1., 0.]])
        boundary = torch.tensor([[3., 2.], [-1., 2.], [3., 3.], [3., 3.]])
        present = torch.tensor([[True, True], [True, True], [True, True], [False, False]])
        indices, margins = compare_name_scores(decision, boundary, present)
        self.assertEqual(indices.tolist(), [0, 0, 0, 0])
        self.assertEqual(margins.tolist(), [1., -1., -1., -torch.inf])

    def test_joint_head_masks_padded_and_empty_candidate_lists(self):
        head = JointNameRoleHead(5)
        features = torch.rand(2, 2, 6, 5)
        present = torch.tensor([[True, False], [False, False]])
        decision, boundary = head(features, present)
        self.assertEqual(decision.shape, (2, 3))
        self.assertTrue(torch.isfinite(decision[:, -1]).all())
        self.assertTrue(torch.isneginf(decision[:, :-1][~present]).all())
        self.assertEqual(boundary.shape, present.shape)

    def test_contact_can_be_present_without_becoming_the_write_target(self):
        subject, contact = (0, 3, 0), (8, 11, 1)
        self.assertEqual(select_name([subject, contact], [1., -.5], threshold=0.), subject)
        self.assertIsNone(select_name([subject, contact], [1., .5], threshold=0.))
        self.assertIsNone(select_name([subject, contact], [0., -1.], threshold=0.))
        self.assertIsNone(select_name([], [], threshold=0.))

    def test_candidate_features_use_original_boundaries(self):
        hidden = torch.eye(5)
        features = mention_features(hidden, 1, 3)
        self.assertTrue(torch.equal(features[1], hidden[0]))
        self.assertTrue(torch.equal(features[2], hidden[3]))
        self.assertTrue(torch.equal(features[3], hidden[4]))
        edge = mention_features(hidden, 0, len(hidden))
        self.assertEqual(int(edge[1:4].count_nonzero()), 0)
        self.assertEqual(NameRoleHead(5, contextual=True)(features.unsqueeze(0)).shape, (1,))
        with self.assertRaises(ValueError):
            mention_features(hidden, 2, 6)
