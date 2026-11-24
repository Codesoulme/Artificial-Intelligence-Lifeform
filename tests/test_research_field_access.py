import unittest

import torch

from spora.research_dialogue_working_state import FIELDS, WorkingTurnState
from spora.research_field_access import (
    EXCLUDE, IGNORE, OPTIONAL, REQUIRE, FieldAccessPlan, MissingFieldActions,
    PairedFieldScopeHead, TokenFieldAccessHead, partial_masks, partial_targets, typed_field_surface,
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


class TypedFieldTests(unittest.TestCase):
    def test_bound_address_keeps_the_exact_value_under_the_same_role(self):
        encode = lambda text: list(map(ord, text))
        name = '取件人'
        value = tuple(encode('辛雨'))
        known = typed_field_surface(encode, name, value)
        unknown = typed_field_surface(encode, name)
        self.assertEqual(''.join(map(chr, known)), '取件人为辛雨。')
        self.assertEqual(''.join(map(chr, unknown)), '取件人尚未记录。')
        self.assertEqual(known[len(name) + 1:-1], value)
        with self.assertRaises(ValueError):
            typed_field_surface(encode, ' ')

    def test_field_access_ignores_padded_question_positions(self):
        torch.manual_seed(6)
        model = TokenFieldAccessHead(6, width=8).eval()
        sequence = torch.randn(1, 4, 6)
        plain = model(sequence, torch.ones(1, 4, dtype=torch.bool))
        extra = torch.cat((sequence, 100. * torch.randn(1, 3, 6)), dim=1)
        mask = torch.tensor([[True, True, True, True, False, False, False]])
        scores = model(extra, mask)
        torch.testing.assert_close(scores, plain)
        scores.square().mean().backward()
        self.assertIsNotNone(model.field_queries.grad)
        self.assertTrue(torch.isfinite(model.field_queries.grad).all())


class PairedFieldScopeTests(unittest.TestCase):
    def test_swapping_descriptors_swaps_directional_predictions_only(self):
        torch.manual_seed(17)
        head = PairedFieldScopeHead(8, width=6)
        order, reverse = torch.randn(3, 8), torch.randn(3, 8)
        first_scores, reverse_scores = head(order, reverse), head(reverse, order)
        torch.testing.assert_close(first_scores, reverse_scores.index_select(-1, head.reverse))
        self.assertEqual(head.reverse.tolist(), [0, 2, 1, 4, 3, 5])


if __name__ == '__main__':
    unittest.main()
