import unittest
import torch
from torch import nn
from spora.research_memory_path_ablation import ablate_memory_paths


class Residual(nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.base = nn.Identity()
        self.increment = increment

    def forward(self, x):
        return self.base(x) + self.increment


class Reader(nn.Module):
    def forward(self, x, matrices=None):
        return x + 8


class MemoryPathAblationTests(unittest.TestCase):
    def test_targeted_bypass_and_restoration_after_error(self):
        reader, low, direct, final = Reader(), Residual(1), Residual(2), Residual(4)
        x = torch.tensor([0.])
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with ablate_memory_paths(reader, [low], [direct], [final], {'native', 'direct'}):
                self.assertEqual(reader(x).item(), 0)
                self.assertEqual(low(x).item(), 1)
                self.assertEqual(direct(x).item(), 0)
                self.assertEqual(final(x).item(), 4)
                raise RuntimeError('stop')
        self.assertEqual(reader(x).item(), 8)
        self.assertEqual(direct(x).item(), 2)

    def test_all_paths_and_invalid_configuration(self):
        reader, low, direct, final = Reader(), Residual(1), Residual(2), Residual(4)
        x = torch.tensor([3.])
        with ablate_memory_paths(reader, [low], [direct], [final], {'native', 'low_rank', 'direct', 'final'}):
            self.assertEqual(final(direct(low(reader(x)))).item(), 3)
        with self.assertRaises(ValueError):
            with ablate_memory_paths(reader, [low], [direct], [final], {'unknown'}):
                pass
        with self.assertRaises(ValueError):
            with ablate_memory_paths(reader, [low], [low], [final], {'direct'}):
                pass

    def test_cross_attention_is_isolated_and_restored(self):
        reader, cross = Reader(), Residual(7)
        x = torch.tensor([2.])
        with ablate_memory_paths(reader, [], [], [], {'cross'}, cross=[cross]):
            self.assertEqual(cross(x).item(), 2)
        self.assertEqual(cross(x).item(), 9)
