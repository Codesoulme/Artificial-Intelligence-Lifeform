import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.spora_event_scan_selector import select_event
from spora.durable import TensorStore
from spora.growing import load, new_state, save
from spora.historical_resolutions import HistoricalResolutions
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA


class HistoricalResolutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4321)
        torch.set_num_threads(2)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                  expert_rank=4, memory_layers=(0, 1),
                                                  history_writer='rls', current_address='orthogonal'))
        self.state = new_state(self.model)
        self.state.cursor = 3
        self.model.historical_resolutions = HistoricalResolutions(self.model.config.key_count, capacity=2)
        self.provenance = {'action': 'defer', 'source': 'direct', 'valid_at': 1}
        self.logits = torch.zeros(1, self.model.config.value_count)
        self.logits[0, 3] = 1

    def record(self, decision='confirm'):
        with patch('spora.historical_resolutions.read_provenance', return_value=self.provenance), \
             patch('spora.historical_resolutions.query', return_value=self.logits):
            self.model.historical_resolutions.record(self.model, self.state, 1, 2,
                                                     value=3, valid_at=1, decision=decision)

    def test_explicit_resolution_roundtrips(self):
        self.record()
        schema, tensors = flatten(self.model, self.state)
        model, state = restore(schema, tensors)
        self.assertEqual(model.historical_resolutions.lookup(1, 2)['decision'], 'confirm')
        model.historical_resolutions.validate(model, state)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'individual.pt'
            save(path, self.model, self.state)
            model, state, _ = load(path, 'cpu')
            self.assertEqual(model.historical_resolutions.lookup(1, 2)['value'], 3)
            model.historical_resolutions.validate(model, state)
            with TensorStore.create(Path(directory) / 'individual.sqlite', self.model, self.state) as store:
                model, state, _, revision = store.read()
                self.assertEqual(revision, 0)
                self.assertEqual(model.historical_resolutions.lookup(1, 2)['decision'], 'confirm')
                model.historical_resolutions.validate(model, state)

    def test_inconsistent_or_duplicate_records_do_not_mutate(self):
        before = self.model.historical_resolutions.records.clone()
        with patch('spora.historical_resolutions.read_provenance', return_value=self.provenance), \
             patch('spora.historical_resolutions.query', return_value=self.logits), self.assertRaises(ValueError):
            self.model.historical_resolutions.record(self.model, self.state, 1, 2,
                                                     value=4, valid_at=1, decision='confirm')
        self.assertTrue(torch.equal(before, self.model.historical_resolutions.records))
        self.record()
        with self.assertRaises(ValueError):
            self.record()

    def test_selector_uses_confirmed_event_and_detects_evidence_drift(self):
        self.record()
        rows = [{'action': 'new', 'source': 'direct', 'valid_at': 0},
                {'action': 'correct', 'source': 'direct', 'valid_at': 2}, self.provenance]
        self.state.internal = SimpleNamespace(start_cursor=0)
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=rows), \
             patch('spora.historical_resolutions.query', return_value=self.logits):
            result = select_event(self.model, self.state, 1, 'effective_previous', as_of=2)
            self.assertEqual((result['status'], result['event_time']), ('experimental_candidate', 2))
            self.logits[0, 4] = 2
            result = select_event(self.model, self.state, 1, 'effective_previous', as_of=2)
            self.assertEqual(result['reason'], 'unresolved_factual_observation')

    def test_rejected_event_is_not_candidate(self):
        self.record('reject')
        rows = [{'action': 'new', 'source': 'direct', 'valid_at': 0},
                {'action': 'correct', 'source': 'direct', 'valid_at': 2}, self.provenance]
        self.state.internal = SimpleNamespace(start_cursor=0)
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=rows), \
             patch('spora.historical_resolutions.query', return_value=self.logits):
            self.assertEqual(select_event(self.model, self.state, 1, 'effective_previous', as_of=2)['event_time'], 0)

    def test_invalid_saved_resolution_cannot_be_flattened(self):
        self.record()
        self.model.historical_resolutions.records[0, 1] = self.state.cursor
        with self.assertRaisesRegex(ValueError, 'unobserved'):
            flatten(self.model, self.state)


if __name__ == '__main__':
    unittest.main()
