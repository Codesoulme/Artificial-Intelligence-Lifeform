import unittest
import torch
from torch import nn
from scripts.deepseek_spora_semantic_bridge_pilot import snapshot_final_adapters


class FakeFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_hidden = nn.Linear(3, 2, bias=False)
        self.gate = nn.Parameter(torch.tensor(-1.))


class FinalRecoveryTests(unittest.TestCase):
    def test_snapshot_restores_final_residual_without_aliasing(self):
        source = FakeFinal()
        saved = snapshot_final_adapters([('final_norm', 'output', source)])
        original = source.to_hidden.weight.detach().clone()
        with torch.no_grad():
            source.to_hidden.weight.add_(2)
            source.gate.add_(3)
        restored = FakeFinal()
        restored.to_hidden.load_state_dict(saved[0]['to_hidden'], strict=True)
        with torch.no_grad():
            restored.gate.copy_(saved[0]['gate'])
        self.assertEqual((saved[0]['location'], saved[0]['name']), ('final_norm', 'output'))
        torch.testing.assert_close(restored.to_hidden.weight, original)
        self.assertEqual(restored.gate.item(), -1.)
