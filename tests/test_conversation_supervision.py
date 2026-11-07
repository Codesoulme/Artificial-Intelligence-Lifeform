import copy
import unittest
import torch
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state
from spora.conversation import ConversationBridge, ConversationConfig
from spora.conversation_optimization import backward_supervised
from spora.conversation_supervision import backward_selected, supervised_logits


class SelectedProjectionTests(unittest.TestCase):
    def test_full_context_attention_keeps_masked_loss_gradients(self):
        torch.set_num_threads(2); torch.manual_seed(231603)
        core = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=64, expert_rank=4, memory_layers=(0,1),
            history_writer='rls', current_address='orthogonal'))
        state = new_state(core)
        with torch.no_grad():
            for matrix in state.current: matrix.normal_(0, .1)
        dense = ConversationBridge(core, ConversationConfig(vocabulary=64,
            width=32, depth=2, heads=4, hidden=48, context=64, memory_layers=(0,1)))
        selected = copy.deepcopy(dense)
        tokens = torch.randint(0,64,(5,24)); labels = torch.randint(0,64,(5,24))
        labels[:2] = -100; labels[2, :12] = -100; labels[3, ::2] = -100
        mask = labels != -100
        with torch.no_grad():
            self.assertTrue(torch.allclose(dense(tokens,state)[mask],
                supervised_logits(selected,state,tokens,mask), atol=2e-6,rtol=2e-5))
        a = backward_supervised(dense,state,tokens,labels,2)
        b = backward_selected(selected,state,tokens,labels,2)
        self.assertAlmostEqual(float(a),float(b),places=6)
        for p,q in zip(dense.trainable_parameters(),selected.trainable_parameters()):
            self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-6,rtol=2e-5))
        # User-only positions still affect later predictions through attention.
        self.assertGreater(float(selected.embedding.weight.grad.abs().sum()),0.)


if __name__ == '__main__': unittest.main()
