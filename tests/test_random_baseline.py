import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn

from spora.conversation import ConversationConfig, generate
from spora.growing import new_state
from spora.random_baseline import StandardCausalModel, make_paired_models
from spora.temporal import TemporalConfig, TemporalSPORA


class NoAccess:
    def __getattribute__(self, name):
        raise AssertionError('standard model inspected state')


class RandomBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.config = ConversationConfig(vocabulary=64, width=32, depth=2, heads=4,
                                         hidden=64, context=32, memory_layers=(0, 1))
        self.organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=64, expert_rank=4, memory_layers=(0, 1),
            history_writer='rls', current_address='orthogonal'))
        self.state = new_state(self.organism)

    def test_paired_backbone_seed_and_frozen_individual(self):
        before = {n: p.detach().clone() for n, p in self.organism.named_parameters()}
        rng = torch.random.get_rng_state().clone()
        bridge, standard = make_paired_models(self.organism, self.config, seed=391109)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, value in standard.state_dict().items():
            self.assertTrue(torch.equal(value, bridge.state_dict()[name]), name)
            self.assertNotEqual(value.data_ptr(), bridge.state_dict()[name].data_ptr())
        for name, value in self.organism.named_parameters():
            self.assertTrue(torch.equal(value, before[name]))
            self.assertFalse(value.requires_grad)
        second, other = make_paired_models(self.organism, self.config, seed=391109)
        for name, value in standard.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[name]), name)
        self.assertTrue(torch.equal(bridge.memory_query[0].weight, second.memory_query[0].weight))
        _, different = make_paired_models(self.organism, self.config, seed=391110)
        self.assertFalse(torch.equal(standard.embedding.weight, different.embedding.weight))
        self.assertTrue(torch.equal(bridge.memory_gate, torch.full((2,), -3.)))

    def test_causality_state_independence_and_real_interface(self):
        bridge, model = make_paired_models(self.organism, self.config, seed=4)
        tokens = torch.randint(0, 64, (2, 12))
        changed = tokens.clone()
        changed[:, 7:] = (changed[:, 7:] + 1) % 64
        with torch.no_grad():
            expected = model(tokens, NoAccess())
            self.assertEqual(expected.shape, (2, 12, 64))
            self.assertEqual(bridge(tokens, self.state).shape, expected.shape)
            self.assertTrue(torch.equal(expected[:, :7], model(changed, None)[:, :7]))
            self.assertTrue(torch.equal(expected, model(tokens, object())))
            first, past = model(tokens[:, :4], NoAccess(), use_past=True)
            middle, past = model(tokens[:, 4:9], object(), past=past, use_past=True)
            last, past = model(tokens[:, 9:], None, past=past, use_past=True)
            torch.testing.assert_close(torch.cat((first, middle, last), 1), expected,
                                       atol=1e-6, rtol=1e-5)
            self.assertIsInstance(generate(model, NoAccess(), [2, 3], 5, 4), dict)
        with torch.no_grad():
            model.embedding.weight.add_(1)
        with self.assertRaises(ValueError):
            model(tokens[:, :1], None, past=past, use_past=True)
        with self.assertRaises(ValueError):
            model(tokens.float(), None)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 33), dtype=torch.long), None)

    def test_bridge_causality_and_memory_gradients(self):
        bridge, model = make_paired_models(self.organism, self.config, seed=18)
        tokens = torch.randint(0, 64, (2, 12))
        changed = tokens.clone()
        changed[:, 7:] = (changed[:, 7:] + 1) % 64
        with torch.no_grad():
            self.assertTrue(torch.equal(bridge(tokens, self.state)[:, :7],
                                        bridge(changed, self.state)[:, :7]))
            with patch.object(bridge, 'read_memory', side_effect=lambda x, state, site: x):
                torch.testing.assert_close(bridge(tokens, self.state), model(tokens), rtol=0, atol=0)
        bridge(tokens, self.state).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in self.organism.parameters()))
        self.assertTrue(all(p.grad is not None for p in bridge.memory_query.parameters()))
        self.assertTrue(all(p.grad is not None for p in bridge.memory_out.parameters()))
        self.assertIsNotNone(bridge.memory_gate.grad)

    def test_parameter_counts_and_tied_output(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(memory_width=90)
                self.extra = nn.Parameter(torch.zeros(7))
        bridge, model = make_paired_models(Dummy(), seed=9)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8_524_032)
        self.assertEqual(sum(p.numel() for p in bridge.trainable_parameters()), 8_616_194)
        self.assertEqual(sum(p.numel() for p in bridge.trainable_parameters()) -
                         sum(p.numel() for p in model.parameters()), 92_162)
        self.assertEqual(len([n for n, _ in model.named_parameters()
                              if n == 'embedding.weight']), 1)
        self.assertFalse(any('output' in n for n, _ in model.named_parameters()))
        with torch.no_grad():
            tokens = torch.tensor([[1, 2, 3]])
            x = model.embedding(tokens)
            for block in model.blocks:
                x, _ = block(x)
            torch.testing.assert_close(model(tokens),
                torch.nn.functional.linear(model.norm(x), model.embedding.weight), rtol=0, atol=0)

    def test_invalid_seed(self):
        for seed in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError):
                make_paired_models(self.organism, self.config, seed=seed)


if __name__ == '__main__':
    unittest.main()
