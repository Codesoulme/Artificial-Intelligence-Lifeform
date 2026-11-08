import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from scripts import p115_encode_stories
from scripts.train_p111_chunk_coverage import make_story_documents
from spora.p115_story_input import validate_candidate
from spora.source_review import file_sha256, verify_training_source_reviews
from spora.story_coverage_run import coverage_contract, record_story, restore_coverage, sample_story


class CandidateInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = self.root/'runs'
        self.runs.mkdir()
        self.token_path = self.root/'tokenizer.json'
        vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        vocab.update({'<|bos|>': 256, '<|eos|>': 257, '<|pad|>': 258})
        tokenizer = Tokenizer(models.BPE(vocab, []))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(['<|bos|>', '<|eos|>', '<|pad|>'])
        tokenizer.save(str(self.token_path))
        self.tokenizer = tokenizer
        self.source = self.root/'input.jsonl'
        self.out = self.runs/'candidate'
        self.run_patch = patch.object(p115_encode_stories, 'RUNS', self.runs)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)

    def create(self, texts=('abc', 'hello', 'x'*511)):
        rows = []
        for index, text in enumerate(texts, 1):
            full, group = p115_encode_stories.identities(text)
            rows.append(dict(text=text, sha256=full, group=group, source_original_sha256=p115_encode_stories.digest(text), source_row_1based=index))
        self.source.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        p115_encode_stories.run(self.source, self.out, True, self.token_path)
        self.token_sha = file_sha256(self.token_path)
        self.summary_sha = file_sha256(self.out/'summary.json')

    def validate(self, length=512):
        return validate_candidate(self.out, self.summary_sha, self.token_sha, length, 259)

    def rewrite_summary(self, update_artifact=None, edit=None):
        path = self.out/'summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        if update_artifact:
            summary['files'][update_artifact] = file_sha256(self.out/update_artifact)
        if edit:
            edit(summary)
        path.write_text(json.dumps(summary), encoding='utf-8')
        self.summary_sha = file_sha256(path)

    def test_p115_to_p111_uniform_sampling_and_coverage(self):
        self.create()
        identity = self.validate()
        self.assertEqual(identity['documents'], 2)
        stream = np.memmap(self.out/'stories.bin', dtype='<u2', mode='r')
        plan = dict(story_sampling_mode='uniform', respect_story_document_boundaries=True,
                    observe_story_coverage=True, length=512, target_shares={'story': 1.0})
        documents = make_story_documents(stream, 256, 257, 258, plan)
        self.assertEqual(documents.starts.tolist(), [0, 5])
        x, y, windows, labels = sample_story(documents, random.Random(7), 4, 512, 'cpu', True)
        coverage = restore_coverage(None, coverage_contract(plan), identity['story_sha256'], documents)
        supervised = record_story(coverage, identity['story_sha256'], windows, labels, y, set())
        self.assertEqual(supervised, int((y != -100).sum()))
        for i, window in enumerate(windows):
            length = window.end-window.start
            self.assertEqual(int(x[i, 0]), 256)
            self.assertEqual(int(y[i, length-1]), 257)
            self.assertTrue((y[i, length:] == -100).all())
        self.assertTrue(coverage.summaries())

    def test_stale_digests_and_inconsistent_content_rejected(self):
        self.create()
        (self.out/'stories.bin').write_bytes((self.out/'stories.bin').read_bytes()+b'\x00\x00')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            self.validate()
        self.rewrite_summary('stories.bin')
        with self.assertRaisesRegex(ValueError, 'trailing'):
            self.validate()

    def test_offset_and_summary_rejected_even_when_rehashed(self):
        self.create()
        path = self.out/'stories.jsonl'
        rows = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
        rows[1]['token_offset'] += 1
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows), encoding='utf-8')
        self.rewrite_summary('stories.jsonl')
        with self.assertRaisesRegex(ValueError, 'offsets'):
            self.validate()

    def test_window_tokenizer_and_summary_identity(self):
        self.create()
        with self.assertRaisesRegex(ValueError, 'sampling'):
            self.validate(4)
        with self.assertRaisesRegex(ValueError, 'tokenizer differs'):
            validate_candidate(self.out, self.summary_sha, '0'*64, 512, 259)
        with self.assertRaisesRegex(ValueError, 'summary changed'):
            validate_candidate(self.out, '0'*64, self.token_sha, 512, 259)
        with self.assertRaisesRegex(ValueError, 'special tokens'):
            validate_candidate(self.out, self.summary_sha, self.token_sha, 512, 258)
        self.rewrite_summary(edit=lambda summary: summary['counts'].__setitem__('rejected', 0))
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.validate()

    def test_boundary_token_inside_text_is_rejected(self):
        self.create(('abc<|bos|>def',))
        with self.assertRaisesRegex(ValueError, 'sampling'):
            self.validate()

    def test_existing_tokenizer_with_synthetic_stories(self):
        self.token_path = p115_encode_stories.TOKENIZER
        self.create(('A synthetic story.', 'Another short example.'))
        identity = validate_candidate(self.out, self.summary_sha, self.token_sha, 512, 8192)
        self.assertEqual(identity['documents'], 2)
        tokenizer = Tokenizer.from_file(str(self.token_path))
        stream = np.memmap(self.out/'stories.bin', dtype='<u2', mode='r')
        plan = dict(story_sampling_mode='uniform', respect_story_document_boundaries=True, length=512)
        documents = make_story_documents(stream, *(tokenizer.token_to_id(s) for s in ('<|bos|>', '<|eos|>', '<|pad|>')), plan)
        x, y, _, _ = sample_story(documents, random.Random(1), 2, 512, 'cpu', True)
        self.assertEqual(x.shape, y.shape)
        self.assertGreater(int((y != -100).sum()), 0)

    def test_review_binds_candidate_binary_and_summary(self):
        self.create()
        self.validate()
        (self.root/'data').mkdir()
        (self.root/'curriculum').mkdir()
        review_path = self.root/'review.json'
        review = dict(approved_for_pilot=True, max_schedule_steps=10,
                      training_source_sha256={'story': file_sha256(self.out/'stories.bin')})
        review_path.write_text(json.dumps(review), encoding='utf-8')
        plan = dict(data='data', curriculum='curriculum', schedule_steps=10, focus_every=11,
                    target_shares={'story': 1}, training_source_reviews={'story': {'path': 'review.json', 'sha256': file_sha256(review_path)}})
        override = dict(path=self.out/'stories.bin', summary_sha256=self.summary_sha)
        with self.assertRaisesRegex(ValueError, 'different candidate summary'):
            verify_training_source_reviews(self.root, plan, override)
        review['candidate_summary_sha256'] = self.summary_sha
        review_path.write_text(json.dumps(review), encoding='utf-8')
        plan['training_source_reviews']['story']['sha256'] = file_sha256(review_path)
        self.assertEqual(verify_training_source_reviews(self.root, plan, override)['story']['artifact_sha256'], file_sha256(self.out/'stories.bin'))
        override['summary_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'different candidate summary'):
            verify_training_source_reviews(self.root, plan, override)


if __name__ == '__main__':
    unittest.main()
