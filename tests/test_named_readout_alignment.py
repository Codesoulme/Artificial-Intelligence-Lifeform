import unittest
from types import SimpleNamespace

import torch

from scripts.deepseek_named_readout_alignment import combined_states


class NamedReadoutAlignmentTests(unittest.TestCase):
    def test_batched_snapshots_are_detached_and_keep_order(self):
        states = [SimpleNamespace(current=[torch.full((1, 3, 3), float(value), requires_grad=True)
                                            for _ in range(2)], pages=[]) for value in (2, 7)]
        combined = combined_states(states)
        self.assertEqual(tuple(combined.current[0].shape), (2, 3, 3))
        self.assertEqual(combined.current[0][:, 0, 0].tolist(), [2.0, 7.0])
        self.assertFalse(combined.current[0].requires_grad)
        self.assertEqual(combined.pages, [])


if __name__ == '__main__':
    unittest.main()
