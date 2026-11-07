import copy
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from spora.conversation import ConversationBridge, ConversationConfig, restore_conversation
from spora.conversation_growth import widen_conversation
from spora.growing import GrowingState, grow, new_state
from spora.language_training import frozen_digest, interface_state
from spora.sleep_memory import SealedPage
from spora.temporal import TemporalConfig, TemporalSPORA
from spora.typed_memory import frozen_value_read


class MemoryFoundationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(25)
        self.core = TemporalSPORA(TemporalConfig(
            width=32, layers=2, heads=2, memory_width=64, expert_rank=4,
            memory_layers=(0, 1), history_writer='rls', current_address='orthogonal'))
        self.state = new_state(self.core)
        self.state.cursor = 1
        self.config = ConversationConfig(vocabulary=64, width=32, depth=2,
            heads=4, hidden=48, context=32, memory_layers=(0, 1), memory_read='typed_v1')

    def test_legacy_snapshot_without_reader_field_restores_exactly(self):
        model = ConversationBridge(self.core, replace(self.config, memory_read='legacy'))
        config = asdict(model.config)
        del config['memory_read']
        payload = dict(format='spora-conversation-v1', organism_sha256='p',
            tokenizer_sha256='v', config=config, state=interface_state(model))
        restored = restore_conversation(copy.deepcopy(self.core), payload, 'p', 'v')
        tokens = torch.tensor([[8, 9, 10]])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(tokens, self.state), restored(tokens, self.state)))
        payload['config']['memory_read'] = 'typed_v1'
        with self.assertRaises(ValueError):
            restore_conversation(self.core, payload, 'p', 'v')

    def test_roles_history_order_and_unused_capacity(self):
        model = ConversationBridge(self.core, self.config)
        state = self.state
        state.current[0] = torch.randn_like(state.current[0])
        state.pages[0][0][0] = torch.randn_like(state.pages[0][0][0])
        swapped = copy.deepcopy(state)
        swapped.current[0], swapped.pages[0][0][0] = swapped.pages[0][0][0], swapped.current[0]
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            a = model.read_memory(x, state, 0)
            self.assertFalse(torch.allclose(a, model.read_memory(x, swapped, 0), atol=1e-6))
            self.assertTrue(torch.equal(a, model.read_memory(x, grow(self.core, state, 3), 0)))
            state = grow(self.core, state)
            state.cursor = self.core.config.time_count + 1
            swapped = copy.deepcopy(state)
            swapped.pages.reverse()
            self.assertFalse(torch.allclose(model.read_memory(x, state, 0),
                model.read_memory(x, swapped, 0), atol=1e-7, rtol=0))
            empty = new_state(self.core)
            q = F.normalize(torch.randn(2, 5, 64), dim=-1)
            self.assertEqual(int(torch.count_nonzero(model.typed_memory(q, empty, 0, 64)[..., 64:])), 0)

    def test_sealed_factor_read_and_query_gradients_match_dense(self):
        q = torch.randn(2, 5, 64, dtype=torch.float64, requires_grad=True)
        left = torch.randn(64, 8, dtype=torch.float64, requires_grad=True)
        right = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)
        dense = frozen_value_read(q, (left @ right).unsqueeze(0))
        factored = frozen_value_read(q, (left, right))
        self.assertTrue(torch.allclose(dense, factored, atol=1e-10, rtol=1e-10))
        gd = torch.autograd.grad(dense.square().sum(), q)[0]
        gf = torch.autograd.grad(factored.square().sum(), q)[0]
        self.assertTrue(torch.allclose(gd, gf, atol=1e-8, rtol=1e-10))
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)
        model = ConversationBridge(self.core, self.config)
        factors = [[left.detach().float(), right.detach().float(), torch.zeros(2080)] for _ in range(2)]
        sealed = GrowingState(self.state.current, [SealedPage(factors, 8)], 64)
        with patch.object(SealedPage, 'materialize', side_effect=AssertionError('Dense expansion forbidden')):
            self.assertTrue(torch.isfinite(model(torch.tensor([[8, 9]]), sealed)).all())

    def test_typed_training_restore_incremental_and_growth(self):
        model = ConversationBridge(self.core, self.config)
        state = grow(self.core, self.state)
        state.cursor = 65  # Two used history slots exercise routing gradients.
        before = frozen_digest(self.core, state)
        tokens = torch.randint(8, 64, (2, 12))
        # Make different histories; newly allocated pages share an initial prior.
        state.pages[1][0][0] = state.pages[1][0][0].detach() + .1
        before = frozen_digest(self.core, state)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=.001)
        logits = model(tokens, state)
        F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()).backward()
        self.assertGreater(float(model.typed_memory.page_keys.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in self.core.parameters()))
        opt.step()
        self.assertEqual(before, frozen_digest(self.core, state))
        payload = dict(format='spora-conversation-v1', organism_sha256='p', tokenizer_sha256='v',
            config=asdict(self.config), state=interface_state(model))
        restored = restore_conversation(copy.deepcopy(self.core), payload, 'p', 'v')
        grown = widen_conversation(model, seed=25)
        with torch.no_grad():
            expected = model(tokens, state)
            self.assertTrue(torch.equal(expected, restored(tokens, state)))
            self.assertTrue(torch.allclose(expected, grown(tokens, state), atol=2e-6, rtol=2e-5))
            _, past = model(tokens[:, :7], state, use_past=True)
            actual, _ = model(tokens[:, 7:], state, past=past, use_past=True)
            self.assertTrue(torch.allclose(expected[:, 7:], actual, atol=1e-6, rtol=1e-5))

    def test_incremental_state_rejects_changed_memory_and_weights(self):
        for kind in ('legacy', 'typed_v1'):
            model = ConversationBridge(self.core, replace(self.config, memory_read=kind))
            tokens = torch.tensor([[8, 9]])
            with torch.no_grad():
                _, past = model(tokens, self.state, use_past=True)
                self.state.current[0].add_(.1)
                with self.assertRaisesRegex(ValueError, 'rebuild'):
                    model(tokens[:, :1], self.state, past=past)
                _, past = model(tokens, self.state, use_past=True)
                model.embedding.weight.add_(.1)
                with self.assertRaisesRegex(ValueError, 'rebuild'):
                    model(tokens[:, :1], self.state, past=past)
                _, past = model(tokens, self.state, use_past=True)
                replacement = copy.deepcopy(self.state)
                with self.assertRaisesRegex(ValueError, 'rebuild'):
                    model(tokens[:, :1], replacement, past=past)
                with self.assertRaisesRegex(ValueError, 'Unbound'):
                    model(tokens[:, :1], self.state, past=list(past))

    def test_invalid_dimensions_fail_before_network_construction(self):
        for key, value in [('heads', 0), ('width', -1), ('hidden', 0), ('memory_read', 'unknown')]:
            with self.assertRaises(ValueError):
                replace(self.config, **{key: value}).validate()

    def test_mid_forward_memory_change_cannot_publish_incremental_state(self):
        model = ConversationBridge(self.core, self.config)
        def mutate(*args):
            with torch.no_grad():
                self.state.current[0].add_(.1)
        hook = model.blocks[0].register_forward_hook(mutate)
        try:
            with torch.no_grad(), self.assertRaisesRegex(ValueError, 'rebuild'):
                model(torch.tensor([[8, 9]]), self.state, use_past=True)
        finally:
            hook.remove()


if __name__ == '__main__':
    unittest.main()
