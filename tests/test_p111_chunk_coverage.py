import copy
import hashlib
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.train_p111_chunk_coverage import (chunk_contract, record_chunk, restore_chunk_coverage,
                                               sample_chunk)
from spora.document_chunk_stream import DocumentChunkStream
from spora.document_stream import DocumentStream
from spora.language_training import atomic_save
from spora.story_coverage_run import (coverage_contract, record_story, restore_coverage,
                                      sample_story, validate_identity)


class ChunkCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tokens = np.array([1, 7, 8, 9, 10, 11, 2, 1, 12, 2, 1, 2], dtype=np.uint16)
        self.digest = hashlib.sha256(self.tokens.tobytes()).hexdigest()
        self.chunks = DocumentChunkStream(self.tokens, 1, 2, 0, 3)
        self.stories = DocumentStream(self.tokens, 1, 2, 0)
        self.plan = dict(pretrain_sampling_mode='chunk', respect_document_boundaries=True,
                         observe_pretrain_coverage=True, observe_story_coverage=True,
                         respect_story_document_boundaries=True, length=3,
                         target_shares={'story': .5, 'pretrain': .5})
        self.contract = chunk_contract(self.plan)
        self.story_contract = coverage_contract(self.plan)
        self.sources = {'entry': 'implementation-hash'}

    def test_boundaries_and_focus(self):
        windows = [self.chunks.window(i) for i in range(int(self.chunks.cumulative[-1]))]
        self.assertEqual([(w.document, w.start, w.end) for w in windows],
                         [(0, 1, 4), (0, 4, 7), (1, 1, 3), (2, 1, 2)])
        rng = random.Random(17)
        x, y, ws, before = sample_chunk(self.chunks, rng, 10, 3, 'cpu')
        expected = sum(w.end - w.start for w in ws[1:])
        coverage = restore_chunk_coverage(None, self.contract, self.digest, self.chunks)
        # Row zero is excluded even when the replacement happens to be identical.
        self.assertEqual(record_chunk(coverage, self.digest, ws, before, y, {0}), expected)
        self.assertEqual(sum(v.total_positions for v in coverage.summaries().values()), expected)
        for row, w in enumerate(ws):
            base = int(self.chunks.starts[w.document])
            count = w.end - w.start
            self.assertEqual(x[row, :count].tolist(), self.tokens[base+w.start-1:base+w.end-1].tolist())
            self.assertEqual(y[row, :count].tolist(), self.tokens[base+w.start:base+w.end].tolist())
            self.assertTrue(torch.all(y[row, count:] == -100))

    def run_segment(self, end, path=None, resume=False):
        rng = random.Random(928)
        chunk = restore_chunk_coverage(None, self.contract, self.digest, self.chunks)
        story = restore_coverage(None, self.story_contract, self.digest, self.stories)
        counts = {}; trace = []; start = 0
        if resume:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            validate_identity(payload, self.story_contract, self.sources, self.digest)
            chunk = restore_chunk_coverage(payload, self.contract, self.digest, self.chunks)
            story = restore_coverage(payload, self.story_contract, self.digest, self.stories)
            rng.setstate(payload['python_rng'])
            counts, trace, start = payload['counts'], payload['trace'], payload['steps']
        for step in range(start, end):
            is_chunk = step % 2 == 0
            if is_chunk:
                x, y, windows, before = sample_chunk(self.chunks, rng, 4, 3, 'cpu')
            else:
                x, y, windows, before = sample_story(self.stories, rng, 4, 3, 'cpu', True)
            replaced = set()
            focus = 0
            if step % 3 == 0:
                x[0] = torch.tensor([1, 14, 0]); y[0] = torch.tensor([14, 2, -100])
                replaced.add(0); focus = 2
            if is_chunk:
                n = record_chunk(chunk, self.digest, windows, before, y, replaced)
                key = 'pretrain_source_targets'
            else:
                n = record_story(story, self.digest, windows, before, y, replaced)
                key = 'story_source_targets'
            self.assertEqual(n, int((y != -100).sum()) - focus)
            counts[key] = counts.get(key, 0) + n
            trace.append((x.tolist(), y.tolist()))
        payload = dict(chunk_coverage_entry='spora-chunk-coverage-v1',
                       chunk_coverage_contract=self.contract, pretrain_source_sha256=self.digest,
                       chunk_coverage=chunk.to_dict(), coverage_entry='spora-story-coverage-v1',
                       coverage_contract=self.story_contract, source_code_sha256=self.sources,
                       story_source_sha256=self.digest, coverage=story.to_dict(), counts=counts,
                       python_rng=rng.getstate(), trace=trace, steps=end)
        restore_chunk_coverage(payload, self.contract, self.digest, self.chunks)
        restore_coverage(payload, self.story_contract, self.digest, self.stories)
        if path: atomic_save(payload, path)
        return payload

    def test_resume_and_separate_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            expected = self.run_segment(15)
            self.run_segment(7, path)
            actual = self.run_segment(15, path, True)
        self.assertEqual(expected, actual)
        self.assertTrue(actual['chunk_coverage']['records'])
        self.assertTrue(actual['coverage']['records'])
        self.assertNotEqual(actual['chunk_coverage'], actual['coverage'])

    def test_reject_bad_state_and_contract(self):
        good = self.run_segment(6)
        for change in ('source', 'digest', 'mode', 'length', 'missing', 'count', 'position', 'story', 'implementation'):
            bad = copy.deepcopy(good)
            if change == 'source': bad['pretrain_source_sha256'] = '0' * 64
            elif change == 'digest': bad['chunk_coverage']['records'][0]['source'] = '0' * 64
            elif change == 'mode': bad['chunk_coverage_contract']['mode'] = 'legacy'
            elif change == 'length': bad['chunk_coverage_contract']['window_length'] += 1
            elif change == 'missing': bad.pop('chunk_coverage')
            elif change == 'count': bad['counts']['pretrain_source_targets'] += 1
            elif change == 'position': bad['chunk_coverage']['records'][0]['intervals'][0][1] = 999
            elif change == 'story': bad.pop('coverage')
            else: bad['source_code_sha256'] = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_identity(bad, self.story_contract, self.sources, self.digest)
                restore_coverage(bad, self.story_contract, self.digest, self.stories)
                restore_chunk_coverage(bad, self.contract, self.digest, self.chunks)
        for field, value in [('observe_story_coverage', False), ('respect_story_document_boundaries', False),
                             ('respect_document_boundaries', False), ('observe_pretrain_coverage', False),
                             ('pretrain_sampling_mode', 'legacy')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                chunk_contract({**self.plan, field: value})


if __name__ == '__main__':
    unittest.main()
