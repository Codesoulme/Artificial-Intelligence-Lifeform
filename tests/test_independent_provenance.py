import tempfile
import unittest
from pathlib import Path

import torch

from spora.durable import TensorStore
from spora.growing import capacity, grow, load, new_state, save
from spora.internal_memory import (ACTIONS, SOURCES, WriteControl, attach, enable_event_key_index,
                                   enable_independent_provenance, observe_sourced)
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA


class IndependentProvenanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8127)
        torch.set_num_threads(2)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                  expert_rank=4, memory_layers=(0, 1),
                                                  history_writer='rls', current_address='orthogonal'))
        self.base = attach(self.model, new_state(self.model), WriteControl())
        with torch.no_grad():
            for parameter in self.model.write_control.parameters():
                parameter.zero_()
            self.model.write_control.network[-1].bias[4] = 20

    def test_independent_update_matches_reference_and_preserves_snapshot(self):
        fresh = enable_event_key_index(self.model, self.base)
        state = enable_independent_provenance(self.model, fresh, ridge=0.003)
        width = self.model.config.memory_width
        precision, matrix = torch.eye(width) / 0.003, torch.zeros(27, width)
        for event, key in enumerate((2, 3, 2, 4, 5, 2)):
            keys = torch.tensor([key])
            times = torch.tensor([event])
            modes = torch.ones_like(keys)
            _, bound = self.model.representation(keys, modes, times)
            address = self.model.memory_address(self.model.memories[-1], bound, keys, modes)[0].detach()
            previous_precision = state.internal.provenance_precision[0].clone()
            previous = state
            state, decision = observe_sourced(self.model, state, key, 3, 'direct', 1.0)
            target = torch.cat((torch.nn.functional.one_hot(torch.tensor(SOURCES.index('direct')), 4),
                                torch.nn.functional.one_hot(torch.tensor(ACTIONS.index(decision['action'])), 5),
                                torch.tensor([1] * 8 + [(event >> i) & 1 for i in range(10)]))).float()
            projected = precision @ address
            gain = projected / (1 + address @ projected)
            matrix = matrix + (target - matrix @ address)[:, None] * gain[None]
            precision = precision - projected[:, None] * gain[None]
            precision = (precision + precision.T) * 0.5
            torch.testing.assert_close(state.internal.provenance[0], matrix, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(state.internal.provenance_precision[0], precision, atol=5e-5, rtol=1e-5)
            self.assertTrue(torch.equal(previous.internal.provenance_precision[0], previous_precision))
        self.assertIsNone(fresh.internal.provenance_precision)
        self.assertEqual(state.internal.event_keys[0][:6].tolist(), [2, 3, 2, 4, 5, 2])

    def test_growth_and_persistence_preserve_precision(self):
        state = enable_independent_provenance(self.model, enable_event_key_index(self.model, self.base))
        for event in range(65):
            if state.cursor == capacity(self.model, state):
                state = grow(self.model, state)
            state, _ = observe_sourced(self.model, state, event % 4, event % 3, 'direct', 1.0)
        self.assertEqual(len(state.internal.provenance_precision), 2)
        schema, tensors = flatten(self.model, state)
        model, restored = restore(schema, tensors)
        for a, b in zip(state.internal.provenance_precision, restored.internal.provenance_precision):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(restored.internal.provenance_ridge, 0.003)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'individual.pt'
            save(path, self.model, state)
            model, restored, _ = load(path, 'cpu')
            self.assertTrue(torch.equal(restored.internal.provenance_precision[1], state.internal.provenance_precision[1]))
            with TensorStore.create(Path(directory) / 'individual.sqlite', self.model, state) as store:
                changed, _ = observe_sourced(self.model, state, 6, 2, 'direct', 1.0)
                store.commit(self.model, changed, 0)
                _, restored, _, revision = store.read()
                self.assertEqual(revision, 1)
                self.assertTrue(torch.equal(restored.internal.provenance_precision[1], changed.internal.provenance_precision[1]))

    def test_invalid_options_and_corrupt_precision_are_rejected(self):
        for ridge in (True, 0.0, -1.0, float('nan'), 2.0):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                enable_independent_provenance(self.model, self.base, ridge)
        state = enable_independent_provenance(self.model, self.base)
        state, _ = observe_sourced(self.model, state, 1, 2, 'direct', 1.0)
        with self.assertRaises(ValueError):
            enable_independent_provenance(self.model, state)
        state.internal.provenance_precision[0][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'precision'):
            flatten(self.model, state)


if __name__ == '__main__':
    unittest.main()
