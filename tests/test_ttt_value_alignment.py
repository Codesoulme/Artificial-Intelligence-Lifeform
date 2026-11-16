import unittest

import torch
from torch import nn

from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import value_alignment_loss


class ValueAlignmentTests(unittest.TestCase):
    def test_alignment_distinguishes_matching_and_swapped_values(self):
        writer = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            writer.weight.copy_(torch.eye(2))
        values = torch.eye(2)
        cache = {'targets': {6: values.clone()}, 'value_embeddings': values}
        matching = value_alignment_loss({6: writer}, cache)
        cache['value_embeddings'] = values.flip(0)
        swapped = value_alignment_loss({6: writer}, cache)
        self.assertAlmostEqual(matching.item(), 0.0, places=6)
        self.assertAlmostEqual(swapped.item(), 1.0, places=6)
        swapped.backward()
        self.assertGreater(writer.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
