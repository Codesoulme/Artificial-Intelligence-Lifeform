import unittest

import torch

from spora.research_entity_field_address import EntityFieldAddress, RecurrentFieldSlots, TokenValue, revise_association


class EntityFieldAddressTests(unittest.TestCase):
    def test_factored_keys_revise_one_entity_field(self):
        head = EntityFieldAddress(2, 2, 2)
        with torch.no_grad():
            head.entity.weight.copy_(torch.eye(2))
            head.field.weight.copy_(torch.eye(2))
        addresses = head(torch.eye(2).repeat_interleave(2, dim=0), torch.eye(2).repeat(2, 1))
        before = torch.arange(8).reshape(4, 2).float()
        memory = torch.zeros_like(before)
        for key, value in zip(addresses, before, strict=True):
            memory = revise_association(memory, key, value)
        new = torch.tensor([21., 43.])
        after = revise_association(memory, addresses[0], new)
        torch.testing.assert_close(addresses @ after, torch.cat((new[None], before[1:])))
        torch.testing.assert_close(addresses @ memory, before)

    def test_unknown_token_value_round_trip_and_rejects_overlength(self):
        codec = TokenValue(152064, capacity=6)
        self.assertEqual(codec.decode(codec.encode([0, 2104, 152063])), [0, 2104, 152063])
        with self.assertRaises(ValueError):
            codec.encode([4] * 6)
        self.assertIsNone(codec.decode(torch.ones(codec.width)))
        self.assertIsNone(codec.decode(torch.zeros(codec.width)))

    def test_routed_revision_preserves_other_slots_and_abstains_on_ties(self):
        slots = RecurrentFieldSlots(2, 2, 2)
        first, second = torch.eye(2)
        slots.allocate(first, torch.tensor([1., 2.]))
        slots.allocate(second, torch.tensor([3., 4.]))
        with self.assertRaisesRegex(ValueError, 'No free'):
            slots.allocate(first, torch.ones(2))
        self.assertIsNone(slots.revise(torch.ones(2), torch.zeros(2)))
        self.assertIsNone(slots.revise(torch.tensor([.51, .49]), torch.zeros(2), gap=.04))
        self.assertEqual(slots.revise(second, torch.tensor([9., 8.]), gap=.04), 1)
        torch.testing.assert_close(slots.values, torch.tensor([[1., 2.], [9., 8.]]))

    def test_routed_can_limit_candidates_to_confirmed_entity(self):
        slots = RecurrentFieldSlots(4, 2, 1)
        for key in ([.4, .8], [.1, .7], [1., 0.], [.3, .9]):
            slots.allocate(torch.tensor(key), torch.zeros(1))
        query = torch.tensor([1., 0.])
        self.assertEqual(slots.select(query), 2)
        self.assertEqual(slots.select(query, eligible=slice(0, 2)), 0)
