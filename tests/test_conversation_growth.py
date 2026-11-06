import unittest
from dataclasses import asdict
import torch
from torch.nn import functional as F
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state
from spora.conversation import ConversationBridge, ConversationConfig, restore_conversation
from spora.conversation_growth import widen_conversation
from spora.language_training import interface_state, frozen_digest


class ConversationGrowthTests(unittest.TestCase):
    def test_tied_output_memory_and_incremental_predictions_survive_growth(self):
        torch.set_num_threads(2)
        torch.manual_seed(89)
        core = TemporalSPORA(TemporalConfig(
            width=32, layers=2, heads=2, memory_width=64, expert_rank=4,
            memory_layers=(0, 1), history_writer='rls', current_address='orthogonal'))
        state = new_state(core)
        # Exercise nonzero reads as well as the language branch.
        with torch.no_grad():
            for matrix in state.current:
                matrix.normal_(0, .2)
        source = ConversationBridge(core, ConversationConfig(
            vocabulary=64, width=32, depth=2, heads=4, hidden=48,
            context=64, memory_layers=(0, 1)))
        old = interface_state(source)
        before = frozen_digest(core, state)
        tokens = torch.randint(0, 64, (2, 19))
        grown = widen_conversation(source, seed=71, noise_std=.003)
        with torch.no_grad():
            expected = source(tokens, state)
            actual = grown(tokens, state)
            self.assertTrue(torch.allclose(expected, actual, atol=2e-6, rtol=2e-5))
            self.assertTrue(torch.equal(expected.argmax(-1), actual.argmax(-1)))
            _, past = grown(tokens[:, :12], state, use_past=True)
            tail, _ = grown(tokens[:, 12:], state, past=past, use_past=True)
            self.assertTrue(torch.allclose(actual[:, 12:], tail, atol=2e-6, rtol=2e-5))
            snapshot = dict(format='spora-conversation-v1', organism_sha256='test-parent',
                            tokenizer_sha256='test-vocabulary', config=asdict(grown.config),
                            state=interface_state(grown))
            restored = restore_conversation(core, snapshot, 'test-parent', 'test-vocabulary')
            self.assertTrue(torch.equal(actual, restored(tokens, state)))
        opt = torch.optim.AdamW(grown.trainable_parameters(), lr=.001)
        loss = F.cross_entropy(grown(tokens, state).flatten(0, 1), tokens.flatten())
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in grown.trainable_parameters()))
        opt.step()
        left, right = grown.embedding.weight.chunk(2, dim=1)
        self.assertFalse(torch.equal(left, right), 'Replicas must be able to diverge')
        self.assertEqual(before, frozen_digest(core, state))
        self.assertTrue(all(torch.equal(v, interface_state(source)[k]) for k, v in old.items()))


if __name__ == '__main__':
    unittest.main()
