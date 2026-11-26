import unittest

import torch

from scripts.audit_spora_role_version_routing import (
    CURRENT, FIRST, SKIP, FactorizedRolePlan, factorized_loss, factorized_predictions,
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
