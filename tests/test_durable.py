import tempfile
import unittest
from pathlib import Path
import torch
from spora.durable import TensorStore, RevisionConflict
from spora.growing import new_state, observe, grow, query
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.state_codec import flatten


class DurableTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(221)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        self.state = observe(self.model, new_state(self.model), 2, 5, 1)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'individual.sqlite'

    def equal(self, model, a, other, b):
        sa, ta = flatten(model, a); sb, tb = flatten(other, b)
        self.assertEqual(sa, sb); self.assertEqual(set(ta), set(tb))
        for name in ta:
            torch.testing.assert_close(ta[name], tb[name], atol=0, rtol=0)

    def test_incremental_roundtrip_and_next_write(self):
        with TensorStore.create(self.path, self.model, self.state) as store:
            changed = observe(self.model, self.state, 2, 9, 1)
            stats = store.commit(self.model, changed, 0)
            self.assertEqual(stats['changed_tensors'], 6)
            other, state, _, rev = store.read()
            self.equal(self.model, changed, other, state)
            a = observe(self.model, changed, 3, 7, 0); b = observe(other, state, 3, 7, 0)
            self.equal(self.model, a, other, b)
            expanded = grow(other, b)
            store.commit(other, expanded, rev)
            newer, final, _, _ = store.read()
            self.equal(other, expanded, newer, final)

    def test_partial_transaction_rolls_back_and_keeps_revision(self):
        with TensorStore.create(self.path, self.model, self.state) as store:
            changed = observe(self.model, self.state, 2, 9, 1)
            def fail(stage):
                if stage == 'after_first_tensor':
                    raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):
                store.commit(self.model, changed, 0, hook=fail)
            model, state, _, revision = store.read()
            self.assertEqual(revision, 0)
            self.equal(self.model, self.state, model, state)
            store.commit(model, observe(model, state, 2, 9, 1), 0)

    def test_concurrent_reader_snapshot_and_stale_writer(self):
        with TensorStore.create(self.path, self.model, self.state) as first, TensorStore(self.path) as second:
            m, s, _, old = second.read()
            changed = observe(self.model, self.state, 2, 9, 1)
            def inspect(stage):
                if stage == 'after_manifest':
                    _, prior, _, rev = second.read()
                    self.assertEqual(rev, 0); self.assertEqual(prior.cursor, 1)
            first.commit(self.model, changed, 0, hook=inspect)
            with self.assertRaises(RevisionConflict):
                second.commit(m, observe(m, s, 3, 8, 1), old)
            # Guessing the newer revision without reloading cannot bypass the
            # stale-writer check and reuse obsolete tensor digests.
            with self.assertRaises(RevisionConflict):
                second.commit(m, observe(m, s, 3, 8, 1), old+1)
            _, latest, _, rev = second.read()
            self.assertEqual(rev, 1); self.assertEqual(latest.cursor, 2)

    def test_corruption_and_nonfinite_are_rejected(self):
        with TensorStore.create(self.path, self.model, self.state) as store:
            changed = observe(self.model, self.state, 2, 9, 1)
            with torch.no_grad():
                changed.current[0].fill_(float('nan'))
            with self.assertRaises(ValueError):
                store.commit(self.model, changed, 0)
            _, prior, _, rev = store.read()
            self.assertEqual(prior.cursor, 1); self.assertEqual(rev, 0)
            store.connection.execute("UPDATE tensors SET body=zeroblob(length(body)) WHERE name='current/0'")
            with self.assertRaises(ValueError):
                store.read()

    def test_snapshot_survives_later_updates_and_checkpoint(self):
        backup = Path(self.temp.name)/'snapshot.sqlite'
        with TensorStore.create(self.path, self.model, self.state) as store:
            store.backup(backup)
            store.commit(self.model, observe(self.model, self.state, 2, 9, 1), 0)
            self.assertEqual(store.checkpoint()[0], 0)
        with TensorStore(backup) as old:
            model, state, _, revision = old.read()
            self.assertEqual(revision, 0)
            self.equal(self.model, self.state, model, state)
        with self.assertRaises(FileExistsError):
            TensorStore.create(backup, self.model, self.state)


if __name__ == '__main__':
    unittest.main()
