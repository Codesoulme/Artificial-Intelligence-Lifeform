import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from spora.continuous_bank_store import bank_revision, commit_banks, read_archived_slots, read_bank_tensor, read_banks
from spora.durable import RevisionConflict


class ContinuousBankStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'banks.sqlite3'
        self.payload = dict(banks={('drink', 1): torch.ones(2, 3)}, pages=[{}], addresses={'one': 0})

    def test_history_is_ordered_by_field_and_committed_with_the_update(self):
        key = (0, 'drink', 1)
        revision, digests, _ = commit_banks(self.path, self.payload, None, {})
        self.assertIsNone(read_archived_slots(self.path, 0, 'drink', revision, 1))
        for entity, value in ((0, 2), (1, 3), (0, 4)):
            previous = self.payload['banks']['drink', 1]
            self.payload['banks']['drink', 1] = torch.full_like(previous, value)
            revision, digests, _ = commit_banks(self.path, self.payload, revision, digests, changed=(key,),
                                                archive=(entity, 'drink', previous))
        for entity, back, value in ((0, 1, 3), (0, 2, 1), (1, 1, 2)):
            record = read_archived_slots(self.path, entity, 'drink', revision, back)
            torch.testing.assert_close(record, torch.full((2, 3), float(value)))
        self.assertIsNone(read_archived_slots(self.path, 0, 'drink', revision, 3))
        self.assertIsNone(read_archived_slots(self.path, 0, 'work', revision, 1))
        with self.assertRaises(RevisionConflict):
            read_archived_slots(self.path, 0, 'drink', revision - 1, 1)

        def interrupt():
            raise OSError('interrupted')

        previous = self.payload['banks']['drink', 1]
        self.payload['banks']['drink', 1] = torch.full_like(previous, 5)
        with self.assertRaises(OSError):
            commit_banks(self.path, self.payload, revision, digests, changed=(key,),
                         archive=(0, 'drink', previous), hook=interrupt)
        self.assertEqual(bank_revision(self.path), revision)
        torch.testing.assert_close(read_bank_tensor(self.path, key, revision, digests[key]), previous)
        record = read_archived_slots(self.path, 0, 'drink', revision, 1)
        torch.testing.assert_close(record, torch.full((2, 3), 3.0))

    def test_only_changed_tensor_is_written_and_stale_commit_is_rejected(self):
        revision, digests, size = commit_banks(self.path, self.payload, None, {})
        self.assertEqual((revision, size), (0, 24))
        with self.assertRaises(RevisionConflict):
            commit_banks(self.path, self.payload, None, {})
        self.assertEqual(bank_revision(self.path), 0)
        self.payload['addresses']['two'] = 64
        revision, digests, size = commit_banks(self.path, self.payload, revision, digests)
        self.assertEqual((revision, size), (1, 0))
        self.payload['pages'][0]['drink'] = torch.full((2, 3), 2.0)
        revision, digests, size = commit_banks(self.path, self.payload, revision, digests)
        self.assertEqual((revision, size), (2, 24))
        self.payload['pages'][0]['drink'] = torch.full((2, 3), 3.0)
        key = (1, 'drink', 1)
        revision, digests, size = commit_banks(self.path, self.payload, revision, digests, changed=(key,))
        self.assertEqual((revision, size), (3, 24))
        with self.assertRaises(RevisionConflict):
            commit_banks(self.path, self.payload, 2, digests, changed=(key,))
        actual, actual_revision, actual_digests = read_banks(self.path)
        self.assertEqual((actual_revision, actual_digests), (revision, digests))
        torch.testing.assert_close(actual['banks']['drink', 1], self.payload['banks']['drink', 1])
        torch.testing.assert_close(actual['pages'][0]['drink'], self.payload['pages'][0]['drink'])

    def test_process_exit_before_commit_leaves_previous_revision_readable(self):
        self.payload['banks']['drink', 1] = torch.ones(1024, 1024)
        commit_banks(self.path, self.payload, None, {})
        script = '''
import os
import sys
from pathlib import Path
from spora.continuous_bank_store import commit_banks, read_banks
path = Path(sys.argv[1])
payload, revision, digests = read_banks(path)
payload['addresses']['two'] = 64
payload['banks']['drink', 1] = payload['banks']['drink', 1] * 3
commit_banks(path, payload, revision, digests, changed=((0, 'drink', 1),), hook=lambda: os._exit(77))
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.path)], timeout=30, check=False)
        self.assertEqual(result.returncode, 77)
        journal = self.path.with_name(self.path.name + '-journal')
        with journal.open('rb') as stream:
            self.assertEqual(stream.read(8), bytes.fromhex('d9d505f920a163d7'))
        actual, _, _ = read_banks(self.path)
        self.assertEqual(bank_revision(self.path), 0)
        self.assertEqual(actual['addresses'], {'one': 0})
        torch.testing.assert_close(actual['banks']['drink', 1], self.payload['banks']['drink', 1])

    def test_failed_creation_leaves_destination_available(self):
        def interrupt():
            raise OSError('interrupted')

        with self.assertRaises(OSError):
            commit_banks(self.path, self.payload, None, {}, hook=interrupt)
        self.assertFalse(self.path.exists())
        self.assertEqual(commit_banks(self.path, self.payload, None, {})[0], 0)

    def test_corrupted_tensor_is_rejected(self):
        commit_banks(self.path, self.payload, None, {})
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                connection.execute('UPDATE bank_tensors SET body=zeroblob(length(body))')
        finally:
            connection.close()
        with self.assertRaisesRegex(ValueError, 'Tensor integrity mismatch'):
            read_banks(self.path)
        _, revision, digests = read_banks(self.path, load_tensors=False)
        key = (0, 'drink', 1)
        with self.assertRaisesRegex(ValueError, 'Tensor integrity mismatch'):
            read_bank_tensor(self.path, key, revision, digests[key])

    def test_lazy_reads_only_the_requested_matrix_at_the_session_revision(self):
        key = (1, 'drink', 1)
        self.payload['pages'][0]['drink'] = torch.full((2, 3), 2.0)
        commit_banks(self.path, self.payload, None, {})
        with patch('spora.continuous_bank_store.unpack', side_effect=AssertionError('Unexpected decode')):
            lazy, revision, digests = read_banks(self.path, load_tensors=False)
        self.assertEqual(lazy['banks'], {('drink', 1): None})
        self.assertEqual(lazy['pages'], [{'drink': None}])
        torch.testing.assert_close(read_bank_tensor(self.path, key, revision, digests[key]),
                                   self.payload['pages'][0]['drink'])
        lazy['pages'][0]['drink'] = torch.full((2, 3), 3.0)
        next_revision, next_digests, size = commit_banks(self.path, lazy, revision, digests, changed=(key,))
        self.assertEqual(size, 24)
        with self.assertRaises(RevisionConflict):
            read_bank_tensor(self.path, key, revision, digests[key])
        torch.testing.assert_close(read_bank_tensor(self.path, key, next_revision, next_digests[key]),
                                   lazy['pages'][0]['drink'])
