import contextvars
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from scripts.deepseek_spora_semantic_bridge_pilot import MemorySlotCrossAttention
from spora.continuous_text_memory import ContinuousTextCodec


class MemorySlotCrossAttentionTests(unittest.TestCase):
    def test_disabled_zero_initialization_mask_and_gradient(self):
        torch.manual_seed(21)
        embedding = torch.randn(12, 4)
        codec = ContinuousTextCodec(embedding, memory_width=8, slot_count=2, seed=7)
        base = nn.Linear(4, 4, bias=False)
        context = contextvars.ContextVar('test_memory_slot', default=None)
        module = MemorySlotCrossAttention(base, context, 8, codec, embedding, slots=2, width=4)
        x = torch.randn(1, 2, 4)
        self.assertTrue(torch.equal(module(x), base(x)))
        packet = codec.encode(torch.tensor([3]))
        address = F.one_hot(torch.tensor([0]), 8).float()
        matrix = packet.reshape(1, 8, 1) * address.reshape(1, 1, 8)
        active = (((matrix,), (matrix,)), (address, address), torch.tensor([True]), torch.tensor([0]))
        token = context.set(active)
        try:
            self.assertTrue(torch.equal(module(x), base(x)))
            module(x).sum().backward()
            self.assertGreater(float(module.out.weight.grad.norm()), 0)
            with torch.no_grad():
                module.out.weight.copy_(torch.eye(4))
            self.assertFalse(torch.equal(module(x), base(x)))
            module.enabled = False
            self.assertTrue(torch.equal(module(x), base(x)))
            module.enabled = True
            snapshot = module.memory_state()
            clone = MemorySlotCrossAttention(nn.Linear(4, 4, bias=False), context, 8, codec, embedding, 2, 4)
            clone.load_memory_state(snapshot)
            self.assertTrue(torch.equal(clone.out.weight, module.out.weight))
            self.assertNotIn('lookup_codes', module.state_dict())
            self.assertNotIn('lookup_embeddings', module.state_dict())
        finally:
            context.reset(token)
        self.assertTrue(torch.equal(module(x), base(x)))
        token = context.set((((matrix,), (matrix,)), (address, address), torch.tensor([False]), torch.tensor([0])))
        try:
            self.assertTrue(torch.equal(module(x), base(x)))
        finally:
            context.reset(token)


if __name__ == '__main__':
    unittest.main()
