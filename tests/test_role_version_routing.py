import unittest

import torch

from scripts.audit_spora_role_version_routing import (
    CURRENT, FIRST, SKIP, ClauseEvidencePlan, FactorizedRolePlan, clause_evidence_loss, clause_targets,
    factorized_loss, factorized_predictions, request_clauses, training_requests,
)


class FactorizedRoleRoutingTests(unittest.TestCase):
    def test_read_intent_and_role_gate_are_both_required(self):
        outputs = (
            torch.tensor([-8., 8., 8.]),
            torch.tensor([[8., 8.], [-8., 8.], [8., 8.]]),
            torch.tensor([[[8., -8.], [-8., 8.]]] * 3),
        )
        decisions = factorized_predictions(outputs, .5, .5).tolist()
        self.assertEqual(decisions, [[SKIP, SKIP], [SKIP, FIRST], [CURRENT, FIRST]])

    def test_unrequested_roles_have_no_version_gradient(self):
        version = torch.zeros((2, 2, 2), requires_grad=True)
        target = torch.tensor([[SKIP, CURRENT], [SKIP, SKIP]])
        loss = factorized_loss((torch.zeros(2), torch.zeros((2, 2)), version), target)
        loss.backward()
        self.assertEqual(int(version.grad[0, 1].count_nonzero()), 2)
        self.assertEqual(int(version.grad[target.eq(SKIP)].count_nonzero()), 0)

    def test_turn_intent_stays_fixed_when_roles_swap(self):
        head = FactorizedRolePlan(8)
        features = torch.randn(4, 2, 8)
        intent, roles, version = head(features)
        swapped_intent, swapped_roles, swapped_version = head(features.flip(1))
        torch.testing.assert_close(intent, swapped_intent)
        torch.testing.assert_close(roles, swapped_roles.flip(1))
        torch.testing.assert_close(version, swapped_version.flip(1))

    def test_clause_evidence_tracks_role_order_and_ignores_padding(self):
        head = ClauseEvidencePlan(8)
        whole, local = torch.randn(1, 2, 8), torch.randn(1, 2, 3, 8)
        allowed = torch.tensor([[True, False, True]])
        intent, roles, version, evidence = head(whole, local, allowed)
        swapped = head(whole.flip(1), local.flip(1), allowed)
        torch.testing.assert_close(intent, swapped[0])
        for original, permuted in zip((roles, version, evidence), swapped[1:], strict=True):
            torch.testing.assert_close(original, permuted.flip(1))
        self.assertTrue(evidence[:, :, 1].isneginf().all())
        changed = local.clone()
        changed[:, :, 1] = torch.randn(1, 2, 8) * 15
        for original, output in zip((intent, roles, version, evidence), head(whole, changed, allowed), strict=True):
            torch.testing.assert_close(original, output)

    def test_clause_annotation_follows_the_mention_not_role_order(self):
        case = next(row for row in training_requests() if row[0].startswith('先报'))
        self.assertEqual(len(request_clauses(case[0])), 2)
        self.assertEqual(clause_targets([case], 2).tolist(), [[[False, True], [True, False]]])
        good = torch.tensor([[[-2., 2.], [2., -2.]]])
        targets = clause_targets([case], 2)
        self.assertLess(clause_evidence_loss(good, targets), clause_evidence_loss(good.flip(2), targets))
