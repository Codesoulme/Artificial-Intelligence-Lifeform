import copy
import unittest
import torch
from torch.nn import functional as F
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state
from spora.conversation import ConversationBridge, ConversationConfig
from spora.conversation_optimization import backward_supervised


class MicrobatchTests(unittest.TestCase):
    def test_unequal_targets_and_partial_last_batch_match_dense_gradient(self):
        torch.set_num_threads(2); torch.manual_seed(71)
        core = TemporalSPORA(TemporalConfig(
            width=32, layers=2, heads=2, memory_width=64, expert_rank=4,
            memory_layers=(0, 1), history_writer='rls', current_address='orthogonal'))
        state = new_state(core)
        dense = ConversationBridge(core, ConversationConfig(
            vocabulary=64, width=32, depth=2, heads=4, hidden=48,
            context=64, memory_layers=(0, 1)))
        micro = copy.deepcopy(dense)
        x = torch.randint(0, 64, (5, 16)); y = torch.randint(0, 64, (5, 16))
        y[0, :] = -100; y[1, 3:] = -100; y[2, 8:] = -100
        a = F.cross_entropy(dense(x, state).flatten(0, 1), y.flatten(), ignore_index=-100)
        a.backward()
        b = backward_supervised(micro, state, x, y, microbatch=2)
        self.assertAlmostEqual(float(a.detach()), float(b), places=6)
        for p, q in zip(dense.trainable_parameters(), micro.trainable_parameters()):
            self.assertTrue(torch.allclose(p.grad, q.grad, atol=2e-6, rtol=2e-5))
        for model in (dense, micro):
            opt = torch.optim.AdamW(model.trainable_parameters(), lr=.0001)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.)
            opt.step()
        for p, q in zip(dense.trainable_parameters(), micro.trainable_parameters()):
            self.assertTrue(torch.allclose(p, q, atol=2e-6, rtol=2e-5))


if __name__ == '__main__':
    unittest.main()
