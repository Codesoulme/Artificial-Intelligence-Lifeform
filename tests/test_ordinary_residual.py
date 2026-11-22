from contextvars import ContextVar
from types import SimpleNamespace
import unittest

import torch
from torch import nn

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

if transformers is not None:
    from spora.integrated_checkpoint import _residual_bundle
    from spora.pretrained_attention import SPORAPretrainedBlock, SPORAResidualBlock

    class IdentityBlock(SPORAPretrainedBlock):
        def __init__(self):
            nn.Module.__init__(self)
            self.hidden_size = 2
            self.self_attn = SimpleNamespace(qkv=nn.Linear(2, 2))

        def forward(self, hidden):
            return hidden


@unittest.skipIf(transformers is None, 'requires optional open-base dependencies')
class OrdinaryResidualTests(unittest.TestCase):
    def setUp(self):
        self.memory = ContextVar('memory', default=None)
        self.hidden = torch.ones(2, 1, 2)

    def block(self, **options):
        block = SPORAResidualBlock(IdentityBlock(), rank=1, **options)
        with torch.no_grad():
            block.down.weight.fill_(1)
            block.up.weight.fill_(1)
        return block

    def test_ordinary_residual_bypasses_entire_memory_call(self):
        block = self.block(memory_context=self.memory, ordinary_only=True)
        self.assertTrue((block(self.hidden) > self.hidden).all())
        token = self.memory.set((None, None, torch.tensor([True, False])))
        try:
            self.assertIs(block(self.hidden), self.hidden)
        finally:
            self.memory.reset(token)
        self.assertTrue((block(self.hidden) > self.hidden).all())
        specs, _ = _residual_bundle({2: block})
        self.assertIs(specs[0]['ordinary_only'], True)
        self.assertNotIn('memory_only', specs[0])

    def test_existing_memory_residual_preserves_row_mask(self):
        block = self.block(memory_context=self.memory)
        self.assertIs(block(self.hidden), self.hidden)
        token = self.memory.set((None, None, torch.tensor([True, False])))
        try:
            output = block(self.hidden)
        finally:
            self.memory.reset(token)
        self.assertTrue((output[0] > self.hidden[0]).all())
        self.assertTrue(torch.equal(output[1], self.hidden[1]))
        specs, _ = _residual_bundle({2: block})
        self.assertIs(specs[0]['memory_only'], True)
        self.assertNotIn('ordinary_only', specs[0])
