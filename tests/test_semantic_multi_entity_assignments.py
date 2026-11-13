import unittest
import torch
from torch import nn

from scripts.deepseek_spora_semantic_bridge_pilot import (TokenCodebookResidualDecoder, TokenCodeDecoder,
                                                        add_value_replay, bridge_optimizer_groups,
                                                        multi_entity_assignments, replace_assignment)
from spora.continuous_text_memory import ContinuousTextCodec


class MultiEntityAssignmentTests(unittest.TestCase):
    def test_supervised_value_is_bound_to_each_target(self):
        for target in (0, 1, 2):
            for value in range(5):
                rows = multi_entity_assignments(target, value, 5)
                self.assertEqual(dict(rows)[target], value)
                self.assertEqual(len(dict(rows)), 3)
                self.assertEqual(len({v for _, v in rows}), 3)

    def test_counterfactual_changes_only_requested_memory(self):
        before = multi_entity_assignments(1, 3, 5)
        after = replace_assignment(before, 1, 4)
        self.assertEqual(dict(after)[1], 4)
        self.assertEqual(dict(after)[0], dict(before)[0])
        self.assertEqual(dict(after)[2], dict(before)[2])
        with self.assertRaises(ValueError):
            replace_assignment(before, 8, 4)

    def test_invalid_targets_and_bounds(self):
        with self.assertRaises(ValueError):
            multi_entity_assignments(3, 0, 5)
        with self.assertRaises(ValueError):
            multi_entity_assignments(1, 5, 5)

    def test_replay_keeps_old_values_and_rejects_held_values(self):
        rows = [{'group': 'train', 'value': 0}, {'group': 'train_multi', 'value': 3}]
        result = add_value_replay(rows, 3, 4, 4)
        self.assertEqual([row['value'] for row in result].count(0), 1)
        self.assertEqual([row['value'] for row in result].count(3), 4)
        with self.assertRaises(ValueError):
            add_value_replay(rows + [{'group': 'held_name', 'value': 3}], 3, 4, 4)
        with self.assertRaises(ValueError):
            add_value_replay(rows + [{'group': 'train', 'value': 4}], 3, 4, 4)

    def test_shared_bridge_is_trainable_once(self):
        bridge = nn.Linear(3, 2)
        adapter = nn.Linear(2, 1)
        groups, params = bridge_optimizer_groups(list(adapter.parameters()), bridge, True, 0.001)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(p.requires_grad for p in bridge.parameters()))
        self.assertEqual(len(params), len(list(adapter.parameters())) + len(list(bridge.parameters())))
        optimizer = torch.optim.SGD(groups, lr=.01)
        before = bridge.weight.detach().clone()
        loss = adapter(bridge(torch.ones(1, 3))).square().sum()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, bridge.weight))
        groups, _ = bridge_optimizer_groups(list(adapter.parameters()), bridge, False, 0.001)
        self.assertEqual(len(groups), 1)
        self.assertTrue(all(not p.requires_grad for p in bridge.parameters()))
        with self.assertRaises(ValueError):
            bridge_optimizer_groups(list(bridge.parameters()), bridge, True, 0.001)

    def test_retrieval_bridge_starts_identical_and_ignores_empty_slots(self):
        torch.manual_seed(71)
        embedding = torch.randn(12, 8)
        codec = ContinuousTextCodec(embedding, memory_width=8, slot_count=2, seed=4)
        plain = TokenCodeDecoder(4, 8)
        bridge = TokenCodebookResidualDecoder(codec, embedding)
        with torch.no_grad():
            bridge.linear.weight.copy_(plain.linear.weight)
        codes = torch.stack((codec.codebook[3], torch.zeros(4))).unsqueeze(0)
        self.assertTrue(torch.equal(bridge(codes), plain(codes)))
        self.assertEqual(set(bridge.state_dict()), {'linear.weight', 'mix'})
        groups, params = bridge_optimizer_groups([], bridge, True, .001, .01)
        self.assertEqual(len(groups), 3)
        loss = bridge(codes)[0, 0, 0]
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in params))
        self.assertEqual(float(bridge(codes)[0, 1].norm()), 0.0)
        with torch.no_grad():
            bridge.mix.fill_(10.0)
            expected = torch.nn.functional.normalize(embedding[3], dim=0)
            self.assertTrue(torch.allclose(bridge(codes)[0, 0], expected, atol=1e-5))
            self.assertEqual(float(bridge(codes)[0, 1].norm()), 0.0)


if __name__ == '__main__':
    unittest.main()
