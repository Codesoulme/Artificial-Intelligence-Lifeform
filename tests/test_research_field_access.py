import unittest

import torch

from spora.research_dialogue_working_state import FIELDS, WorkingTurnState
from spora.research_field_access import (
    EXCLUDE, IGNORE, OPTIONAL, REQUIRE, FieldAccessPlan, MissingFieldActions,
    partial_masks, partial_targets,
)
from spora.research_span_action import SpanActionHead


class FieldAccessTests(unittest.TestCase):
    def test_excluded_value_never_survives_projection_and_optional_absence_is_allowed(self):
        required, excluded, optional = (FIELDS.index(field) for field in ('time', 'person', 'place'))
        roles = [IGNORE] * len(FIELDS)
        roles[required], roles[excluded], roles[optional] = REQUIRE, EXCLUDE, OPTIONAL
        plan = FieldAccessPlan(tuple(roles))
        state = WorkingTurnState().update({'person': (4, 8), 'place': (9,)})
        picked = plan.project(state)
        self.assertFalse(picked.slots[excluded])
        self.assertEqual(picked.slots[optional], (9,))
        self.assertEqual(plan.missing(picked), (required,))
        complete = WorkingTurnState().update({'time': (10,)})
        self.assertEqual(plan.missing(plan.project(complete)), ())

    def test_missing_action_disables_consumed_markers_and_preserves_probability_mass(self):
        count = len(FIELDS)
        address = FIELDS.index('time')
        state = WorkingTurnState().update({'place': (3, 5)})
        _, targets = partial_targets([(count + address, (7, 9)), (None, (2,))], 32, 4)
        allowed, missing = partial_masks(state, (address,), targets, 32, 'cpu')
        self.assertEqual(missing[:, address].tolist(), [True, False, False])
        decoder = SpanActionHead(8, width=4)
        adapter = MissingFieldActions(8, width=4)
        key = decoder.field_keys(state, torch.nn.Embedding(32, 8))
        logits = adapter(torch.randn(3, 8), key, allowed, missing, decoder, torch.nn.Linear(8, 32))
        torch.testing.assert_close(logits.exp().sum(dim=-1), torch.ones(3), atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(logits[0, 32 + count + address]))
        self.assertTrue(torch.isneginf(logits[1:, 32 + count:]).all())
        with self.assertRaises(ValueError):
            partial_masks(state, (), targets, 32, 'cpu')


if __name__ == '__main__':
    unittest.main()
