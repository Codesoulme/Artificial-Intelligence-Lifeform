import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.spora_event_scan_selector import select_event
from spora.durable import RevisionConflict, TensorStore
from spora.growing import grow, load, new_state, save
from spora.internal_memory import WriteControl, attach, enable_event_key_index, observe_sourced
from spora.state_codec import flatten, restore
from spora.temporal import TemporalConfig, TemporalSPORA


class EventKeyIndexTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7751)
        torch.set_num_threads(2)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                                  expert_rank=4, memory_layers=(0, 1),
                                                  history_writer='rls', current_address='orthogonal'))
        self.base = attach(self.model, new_state(self.model), WriteControl())
        with torch.no_grad():
            for param in self.model.write_control.parameters():
                param.zero_()
            self.model.write_control.network[-1].bias[4] = 20

    def writes(self):
        state = enable_event_key_index(self.model, self.base)
        snapshots = [state]
        for key in (2, 3, 2):
            state, _ = observe_sourced(self.model, state, key, 1, 'direct', 1.0)
            snapshots.append(state)
        return state, snapshots

    def test_index_is_opt_in_and_updates_without_mutating_previous_state(self):
        state, snapshots = self.writes()
        self.assertIsNone(self.base.internal.event_keys)
        self.assertTrue(torch.equal(snapshots[0].internal.event_keys[0],
                                    torch.full_like(snapshots[0].internal.event_keys[0], -1)))
        self.assertEqual(snapshots[1].internal.event_keys[0][:3].tolist(), [2, -1, -1])
        self.assertEqual(state.internal.event_keys[0][:3].tolist(), [2, 3, 2])
        with self.assertRaises(ValueError):
            enable_event_key_index(self.model, state)

    def test_selector_scans_only_exact_indexed_events(self):
        state, _ = self.writes()
        rows = [{'action': 'new', 'source': 'direct', 'valid_at': 0},
                {'action': 'correct', 'source': 'direct', 'valid_at': 2}]
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=rows) as read:
            result = select_event(self.model, state, 2, 'effective_previous', as_of=2)
        self.assertEqual(result['event_time'], 0)
        self.assertEqual(result['checked_events'], 2)
        self.assertEqual(result['history_events'], 3)
        read.assert_called_once_with(self.model, state, [(2, 0), (2, 2)], 16)

    def test_selector_handles_no_indexed_events(self):
        state, _ = self.writes()
        with patch('scripts.spora_event_scan_selector.read_provenance_batch', return_value=[]) as read:
            result = select_event(self.model, state, 7, 'effective_previous', as_of=2)
        self.assertEqual(result['reason'], 'insufficient_effective_candidates')
        read.assert_called_once_with(self.model, state, [], 16)

    def test_growth_and_serialization_preserve_index(self):
        state, _ = self.writes()
        state = grow(self.model, state)
        schema, tensors = flatten(self.model, state)
        model, restored = restore(schema, tensors)
        self.assertEqual(restored.internal.event_keys[0][:3].tolist(), [2, 3, 2])
        self.assertTrue(bool((restored.internal.event_keys[1] == -1).all()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'individual.pt'
            save(path, self.model, state)
            model, restored, _ = load(path, 'cpu')
            self.assertEqual(restored.internal.event_keys[0][:3].tolist(), [2, 3, 2])
            with TensorStore.create(Path(directory) / 'individual.sqlite', self.model, state) as store:
                changed, _ = observe_sourced(self.model, state, 4, 2, 'direct', 1.0)
                store.commit(self.model, changed, 0)
                _, restored, _, revision = store.read()
                self.assertEqual(revision, 1)
                self.assertEqual(restored.internal.event_keys[0][:4].tolist(), [2, 3, 2, 4])

    def test_missing_or_out_of_range_key_is_rejected(self):
        state, _ = self.writes()
        state.internal.event_keys[0][1] = -1
        with self.assertRaisesRegex(ValueError, 'index'):
            flatten(self.model, state)
        state.internal.event_keys[0][1] = self.model.config.key_count
        with self.assertRaisesRegex(ValueError, 'index'):
            select_event(self.model, state, 2, 'effective_previous', as_of=2)

    def test_failed_commit_and_stale_writer_do_not_publish_index(self):
        state, _ = self.writes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'individual.sqlite'
            with TensorStore.create(path, self.model, state) as first, TensorStore(path) as second:
                stale_model, stale_state, _, old_revision = second.read()
                changed, _ = observe_sourced(self.model, state, 4, 2, 'direct', 1.0)

                def fail(stage):
                    if stage == 'after_manifest':
                        raise RuntimeError('injected interruption')

                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    first.commit(self.model, changed, 0, hook=fail)
                _, prior, _, revision = second.read()
                self.assertEqual(revision, 0)
                self.assertEqual(prior.internal.event_keys[0][:4].tolist(), [2, 3, 2, -1])
                first.commit(self.model, changed, 0)
                stale_changed, _ = observe_sourced(stale_model, stale_state, 5, 2, 'direct', 1.0)
                with self.assertRaises(RevisionConflict):
                    second.commit(stale_model, stale_changed, old_revision)
                _, latest, _, revision = second.read()
                self.assertEqual(revision, 1)
                self.assertEqual(latest.internal.event_keys[0][:4].tolist(), [2, 3, 2, 4])


if __name__ == '__main__':
    unittest.main()
