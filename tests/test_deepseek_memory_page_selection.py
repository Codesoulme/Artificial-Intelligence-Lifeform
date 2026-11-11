import unittest
from types import SimpleNamespace

import torch
from torch import nn

from scripts.deepseek_spora_memory_answer_distill import _memory_args


class FakeOrganism(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(time_count=64)
        self.memories = [SimpleNamespace(current_basis=torch.eye(4)) for _ in range(2)]

    def representation(self, keys, modes, times):
        return None, times.float()[:, None]

    def memory_address(self, site, bound, keys, modes):
        return bound


class MemoryPageSelectionTests(unittest.TestCase):
    def test_global_event_selects_page_and_local_time(self):
        organism = FakeOrganism()
        state = SimpleNamespace(cursor=130, pages=[None, None, None])
        for event, slot, local in ((0, 1, 0), (63, 1, 63), (64, 2, 0), (129, 3, 1)):
            with self.subTest(event=event):
                result = _memory_args(organism, state, {'key': 1, 'intent': 'past', 'event_time': event})
                self.assertEqual(result['memory_slots'].tolist(), [slot])
                self.assertEqual(result['memory_addresses'][0].tolist(), [[float(local)]])
        current = _memory_args(organism, state, {'key': 1, 'intent': 'current'})
        self.assertEqual(current['memory_slots'].tolist(), [0])

    def test_future_negative_and_unallocated_events_are_rejected(self):
        organism = FakeOrganism()
        state = SimpleNamespace(cursor=130, pages=[None, None])
        for event in (-1, 130, True, 128):
            with self.subTest(event=event), self.assertRaises(ValueError):
                _memory_args(organism, state, {'key': 1, 'intent': 'past', 'event_time': event})


if __name__ == '__main__':
    unittest.main()
