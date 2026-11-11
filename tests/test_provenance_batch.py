import unittest

import torch
from spora.growing import grow, new_state
from spora.internal_memory import (WriteControl, attach, observe_sourced,
                                   read_provenance, read_provenance_batch)
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA


class ProvenanceBatchTests(unittest.TestCase):
    def test_matches_scalar_across_pages_reload_and_unrecorded_times(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                              expert_rank=4, memory_layers=(0, 1), history_writer='rls',
                              current_address='orthogonal'))
        state = attach(model, new_state(model), WriteControl())
        for i in range(65):
            if i == model.config.time_count:
                state = grow(model, state)
            state, _ = observe_sourced(model, state, i % 8, i % 16, 'direct', 1.)
        pairs = [(0, 0), (3, 3), (0, 64), (0, 63), (55, 1), (2, 65), (0, 999), (0, 0)]
        expected = [read_provenance(model, state, key, time) for key, time in pairs]
        schema, tensors = flatten(model, state)
        for size in (1, 3, 16):
            self.assertEqual(read_provenance_batch(model, state, pairs, size), expected)
        restored_model, restored_state = restore(schema, tensors)
        self.assertEqual(read_provenance_batch(restored_model, restored_state, pairs, 3), expected)
        after_schema, after_tensors = flatten(model, state)
        self.assertEqual(schema, after_schema)
        for name in tensors:
            self.assertTrue(torch.equal(tensors[name], after_tensors[name]), name)

    def test_rejects_invalid_input(self):
        model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                              expert_rank=4, memory_layers=(0, 1), history_writer='rls',
                              current_address='orthogonal'))
        state = attach(model, new_state(model), WriteControl())
        self.assertEqual(read_provenance_batch(model, state, []), [])
        for pairs, size in [([(True, 0)], 1), ([(0, -1)], 1), ([(10000, 0)], 1),
                            ([(0, 0)], 0), ([(0, 0)], 65), ([(0, 0)], True)]:
            with self.subTest(pairs=pairs, size=size), self.assertRaises(ValueError):
                read_provenance_batch(model, state, pairs, size)


if __name__ == '__main__':
    unittest.main()
