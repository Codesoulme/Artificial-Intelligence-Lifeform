from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from spora.research_ttt_ntp import LowRankFastWeight
from spora.research_version_store import MemoryAddress, VersionStore


class VersionStoreTests(unittest.TestCase):
    def test_independent_conditions_and_idempotent_commit_survive_reopen(self):
        weekday = MemoryAddress('安若帆', '饮品', '工作日')
        weekend = MemoryAddress('安若帆', '饮品', '周末')
        first = date(2026, 10, 2)
        later = date(2026, 10, 9)
        state = LowRankFastWeight(torch.eye(2), torch.eye(2))
        correction = LowRankFastWeight(torch.eye(2), torch.ones((2, 2)))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'versions.sqlite3'
            with VersionStore(path, 'binding', create=True) as store:
                original = store.append('first', weekday, first, first, state)
                other = store.append('weekend', weekend, first, first, state)
            with VersionStore(path, 'binding') as store:
                new = store.append('correction', weekday, later, later, correction)
                self.assertEqual(store.append('correction', weekday, later, later, correction), new)
                with self.assertRaises(ValueError):
                    store.append('correction', weekday, later, later, state)
                self.assertEqual([entry.number for entry in store.catalog()[weekday]], [original, new])
                self.assertEqual([entry.number for entry in store.catalog()[weekend]], [other])
                self.assertTrue(torch.equal(store.read(original).values, state.values))
                self.assertTrue(torch.equal(store.read(new).values, correction.values))
            with self.assertRaises(ValueError):
                VersionStore(path, 'wrong model')
