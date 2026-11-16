import tempfile
import subprocess
import sys
import copy
from unittest.mock import patch
import unittest
from pathlib import Path

import torch

from spora.durable import RevisionConflict
from spora.research_chunk_history_store import MemoryIntegrityError, ResearchChunkHistoryStore
from spora.structured_text_memory import StructuredTextMemoryCodec
from spora.temporal import TemporalConfig, TemporalSPORA


class ResearchChunkHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(991)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'research.sqlite'
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=32,
                    key_count=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls',
                    current_address='orthogonal'))
        embedding = torch.randn((32, 64), generator=torch.Generator().manual_seed(42))
        self.codec = StructuredTextMemoryCodec(embedding, memory_width=32, identity_slots=2,
                    fact_slots=2, identity_slot_width=8, fact_slot_width=8, eos_token_id=0)

    def create(self):
        store = ResearchChunkHistoryStore.create(self.path, self.model, self.codec)
        self.addCleanup(store.close)
        return store

    def test_transaction_reopen_current_and_history(self):
        store = self.create()
        self.assertEqual(store.write(1, [0, 1], [1], [2, 3])['times'], [0, 1])
        with self.assertRaisesRegex(ValueError, 'explicit update'):
            store.write(1, [0, 1], [1], [4, 5])
        store.write(1, [0, 1], [1], [4, 5], allow_update=True)
        reopened = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.read(1), {'identity_ids': [1], 'fact_ids': [4, 5]})
        self.assertEqual(reopened.read(1, 0), {'identity_ids': [1], 'fact_ids': [2, 3]})
        self.assertEqual(reopened.read(1, 1), {'identity_ids': [1], 'fact_ids': [4, 5]})
        self.assertIsNone(reopened.read(2))
        self.assertEqual(reopened.state.cursor, 4)

    def test_failed_commit_and_stale_writer(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        stale = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(stale.close)
        def fail(stage):
            if stage == 'after_first_tensor':
                raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            store.write(1, [0, 1], [1], [4, 5], allow_update=True, hook=fail)
        self.assertEqual(store.state.cursor, 2)
        self.assertEqual(store.read(1)['fact_ids'], [2, 3])
        store.write(1, [0, 1], [1], [4, 5], allow_update=True)
        with self.assertRaises(RevisionConflict):
            stale.write(2, [2, 3], [2], [6, 7])
        check = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(check.close)
        self.assertEqual(check.read(1)['fact_ids'], [4, 5])
        self.assertIsNone(check.read(2))

    def test_reject_identity_collision_and_wrong_codec(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            store.write(1, [0, 1], [2], [4, 5], allow_update=True)
        with self.assertRaisesRegex(ValueError, 'another record'):
            store.write(2, [1, 2], [2], [4, 5])
        other = StructuredTextMemoryCodec(torch.randn((32, 64)), memory_width=32, identity_slots=2,
                    fact_slots=2, identity_slot_width=8, fact_slot_width=8, eos_token_id=0)
        with self.assertRaises(ValueError):
            ResearchChunkHistoryStore.open(self.path, other)

    def test_process_kill_before_and_after_commit(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        child = '''
import os, sys, torch
from spora.research_chunk_history_store import ResearchChunkHistoryStore
from spora.structured_text_memory import StructuredTextMemoryCodec
torch.set_num_threads(2)
embedding = torch.randn((32, 64), generator=torch.Generator().manual_seed(42))
codec = StructuredTextMemoryCodec(embedding, memory_width=32, identity_slots=2,
    fact_slots=2, identity_slot_width=8, fact_slot_width=8, eos_token_id=0)
store = ResearchChunkHistoryStore.open(sys.argv[1], codec)
def terminate(stage):
    if stage == sys.argv[2]: os._exit(77)
store.write(1, [0, 1], [1], [4, 5], allow_update=True, hook=terminate)
'''
        for stage, expected, cursor in (('after_first_tensor', [2, 3], 2),
                                        ('after_commit', [4, 5], 4)):
            result = subprocess.run([sys.executable, '-c', child, str(self.path), stage],
                                    capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 77, result.stderr)
            reopened = ResearchChunkHistoryStore.open(self.path, self.codec)
            try:
                self.assertEqual(reopened.state.cursor, cursor)
                self.assertEqual(reopened.read(1)['fact_ids'], expected)
                self.assertEqual(reopened.read(1, 0)['fact_ids'], [2, 3])
            finally:
                reopened.close()

    def test_exception_after_commit_updates_in_process_state(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        def fail(stage):
            if stage == 'after_commit':
                raise RuntimeError('after commit')
        with self.assertRaises(RuntimeError):
            store.write(1, [0, 1], [1], [4, 5], allow_update=True, hook=fail)
        self.assertEqual(store.state.cursor, 4)
        self.assertEqual(store.read(1)['fact_ids'], [4, 5])
        store.write(1, [0, 1], [1], [6, 7], allow_update=True)
        self.assertEqual(store.read(1, 0)['fact_ids'], [2, 3])
        self.assertEqual(store.read(1, 1)['fact_ids'], [4, 5])
        self.assertEqual(store.read(1)['fact_ids'], [6, 7])
        with self.assertRaises(ValueError):
            store.read(True)

    def test_variable_length_versions_reserve_addresses_and_restore(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        store.write(1, [0], [1], [4], allow_update=True)
        with self.assertRaisesRegex(ValueError, 'another record'):
            store.write(2, [1], [2], [6])
        store.write(1, [0, 1, 2], [1], [5, 6, 7], allow_update=True)
        reopened = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(reopened.close)
        for version, values in enumerate(([2, 3], [4], [5, 6, 7])):
            self.assertEqual(reopened.read(1, version)['fact_ids'], values)
        self.assertEqual(reopened.read(1)['fact_ids'], [5, 6, 7])
        with self.assertRaisesRegex(ValueError, 'order'):
            reopened.write(1, [2, 1], [1], [8, 9], allow_update=True)

    def test_failed_expansion_keeps_old_addresses_and_cursor(self):
        store = self.create()
        store.write(1, [0], [1], [2])
        def fail(stage):
            if stage == 'after_first_tensor':
                raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            store.write(1, [0, 1], [1], [3, 4], allow_update=True, hook=fail)
        reopened = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.state.cursor, 1)
        self.assertEqual(reopened.read(1)['fact_ids'], [2])
        reopened.write(2, [1], [2], [5])
        self.assertEqual(reopened.read(2)['fact_ids'], [5])

    def test_fixed_length_legacy_upgrade_is_transactional(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        metadata = copy.deepcopy(store.metadata)
        metadata['format'] = 'spora-research-chunk-history-v1'
        store.store.commit(store.model, store.state, expected_revision=store.store.revision, metadata=metadata)
        store.metadata = metadata
        legacy = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(legacy.close)
        self.assertEqual(legacy.read(1)['fact_ids'], [2, 3])
        def fail(stage):
            if stage == 'before_commit':
                raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            legacy.write(1, [0], [1], [4], allow_update=True, hook=fail)
        check = ResearchChunkHistoryStore.open(self.path, self.codec)
        try:
            self.assertEqual(check.metadata['format'], 'spora-research-chunk-history-v1')
            self.assertEqual(check.read(1)['fact_ids'], [2, 3])
        finally:
            check.close()
        legacy.write(1, [0], [1], [4], allow_update=True)
        self.assertEqual(legacy.metadata['format'], 'spora-research-chunk-history-v2')
        self.assertEqual(legacy.read(1, 0)['fact_ids'], [2, 3])
        self.assertEqual(legacy.read(1)['fact_ids'], [4])

    def test_history_drift_refuses_commit_and_preserves_snapshot(self):
        store = self.create()
        store.write(1, [0, 1], [1], [2, 3])
        original_read = ResearchChunkHistoryStore._read_unchecked
        def corrupt_candidate(view, record_id, version=None):
            result = original_read(view, record_id, version)
            if view.state.cursor > 2 and record_id == 1 and version == 0:
                return {'identity_ids': result['identity_ids'], 'fact_ids': [7, 7]}
            return result
        with patch.object(ResearchChunkHistoryStore, '_read_unchecked', corrupt_candidate):
            with self.assertRaisesRegex(MemoryIntegrityError, 'prior historical'):
                store.write(1, [0, 1], [1], [4, 5], allow_update=True)
        self.assertEqual(store.state.cursor, 2)
        self.assertEqual(store.store.revision, 1)
        reopened = ResearchChunkHistoryStore.open(self.path, self.codec)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.read(1)['fact_ids'], [2, 3])
        self.assertEqual(reopened.read(1, 0)['fact_ids'], [2, 3])
