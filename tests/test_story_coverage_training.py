import copy
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from spora.document_stream import DocumentStream
from spora.language_training import atomic_save
from spora.story_coverage_run import (coverage_contract, restore_coverage, sample_story,
                                      record_story, validate_identity)


class StoryCoverageTrainingTest(unittest.TestCase):
    def setUp(self):
        self.data = np.array([1, 4, 5, 6, 2, 1, 7, 8, 9, 10, 2], dtype=np.uint16)
        self.source = hashlib.sha256(self.data.tobytes()).hexdigest()
        self.docs = DocumentStream(self.data, 1, 2, 0)
        self.fingerprints = {'new-entry': 'abc'}

    def contract(self, enabled=True, boundary=True):
        return coverage_contract({'observe_story_coverage': enabled,
                                  'respect_story_document_boundaries': boundary,
                                  'target_shares': {'story': .8, 'dialogue': .2}})

    def run_segment(self, enabled, end, path=None, resume=False):
        contract = self.contract(enabled)
        torch.manual_seed(12)
        rng = random.Random(35)
        model = nn.Sequential(nn.Embedding(16, 8), nn.Dropout(.2), nn.Linear(8, 16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        counts = {}
        coverage = restore_coverage(None, contract, self.source, self.docs)
        start = 0
        trace = []
        if resume:
            payload = torch.load(path, weights_only=True)
            validate_identity(payload, contract, self.fingerprints, self.source)
            coverage = restore_coverage(payload, contract, self.source, self.docs)
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            rng.setstate(payload['python_rng'])
            torch.set_rng_state(payload['torch_rng'])
            counts, start, trace = payload['counts'], payload['steps'], payload['trace']
        for step in range(start, end):
            x, y, windows, sampled = sample_story(self.docs, rng, 3, 4, 'cpu', contract['record_story'])
            replaced = set()
            if step % 2 == 0:
                x[0] = torch.tensor([1, 11, 12, 0])
                y[0] = torch.tensor([11, 12, -100, -100])
                replaced.add(0)
            # Mask a surviving target, testing accounting after the final edit.
            y[1, 0] = -100
            count = int((y != -100).sum()) - (2 if replaced else 0)
            observed = record_story(coverage, self.source, windows, sampled, y, replaced)
            if enabled:
                self.assertEqual(observed, count)
            counts['story_source_targets'] = counts.get('story_source_targets', 0) + count
            trace.append((x.tolist(), y.tolist()))
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x).reshape(-1, 16), y.flatten(), ignore_index=-100)
            loss.backward()
            optimizer.step()
        payload = dict(coverage_entry='spora-story-coverage-v1', coverage_contract=contract,
                       source_code_sha256=self.fingerprints, story_source_sha256=self.source,
                       coverage=coverage.to_dict(), counts=counts, model=model.state_dict(),
                       optimizer=optimizer.state_dict(), python_rng=rng.getstate(),
                       torch_rng=torch.get_rng_state(), steps=end, trace=trace)
        restore_coverage(payload, contract, self.source, self.docs)
        if path:
            atomic_save(payload, path)
        return payload

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_equivalence_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            enabled = self.run_segment(True, 8)
            disabled = self.run_segment(False, 8)
            self.run_segment(True, 3, path)
            resumed = self.run_segment(True, 8, path, True)
            self.assert_nested_equal(enabled, resumed)
            for key in ('model', 'optimizer', 'python_rng', 'torch_rng', 'trace', 'counts'):
                self.assert_nested_equal(enabled[key], disabled[key])
            self.assertTrue(enabled['coverage']['records'])
            self.assertEqual(disabled['coverage']['records'], [])

    def test_exact_positions_after_replacement_and_mask(self):
        rng = random.Random(4)
        x, y, windows, before = sample_story(self.docs, rng, 3, 4, 'cpu', True)
        # Even an identical replacement is excluded by row identity.
        y[1, 0] = -100
        expected = set()
        expected_count = 0
        for row, window in enumerate(windows):
            if row == 0:
                continue
            for column in range(window.end - window.start):
                if y[row, column] != -100:
                    expected.add((window.document, window.start + column))
                    expected_count += 1
        coverage = restore_coverage(None, self.contract(), self.source, self.docs)
        self.assertEqual(record_story(coverage, self.source, windows, before, y, {0}), expected_count)
        actual = set()
        for (_, document), summary in coverage.summaries().items():
            for start, end in summary.intervals:
                actual.update((document, position) for position in range(start, end))
        self.assertEqual(actual, expected)

    def test_scope_and_rejection(self):
        contract = self.contract(True, False)
        self.assertFalse(contract['record_story'])
        self.assertEqual(contract['scope'], 'unavailable_story_boundaries_disabled')
        self.assertIn('dialogue', contract['excluded_sources'])
        self.assertIn('focus_replacements', contract['excluded_sources'])
        self.assertFalse(contract['full_training_coverage'])
        good = self.run_segment(True, 3)
        for change in ('identity', 'source', 'switch', 'state', 'count', 'position'):
            bad = copy.deepcopy(good)
            if change == 'identity': bad.pop('coverage_entry')
            elif change == 'source': bad['story_source_sha256'] = '0' * 64
            elif change == 'switch': bad['coverage_contract']['enabled'] = False
            elif change == 'state': bad.pop('coverage')
            elif change == 'count': bad['counts']['story_source_targets'] += 1
            else: bad['coverage']['records'][0]['intervals'][0][1] = 999
            with self.assertRaises(ValueError, msg=change):
                validate_identity(bad, self.contract(), self.fingerprints, self.source)
                restore_coverage(bad, self.contract(), self.source, self.docs)


if __name__ == '__main__':
    unittest.main()
