import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from spora.document_stream import DocumentStream
from spora.growing import new_state
from spora.hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer
from spora.language_bridge import BOS, EOS, PAD
from spora.language_training import ByteTrainer, atomic_save, frozen_digest, interface_state
from spora.packed_hybrid_training import PackedHybridTrainer
from spora.packed_coverage import PackedCoverage
from spora.temporal import TemporalConfig, TemporalSPORA


class PackedHybridTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        self.state = new_state(self.organism)
        self.vocab = HybridTokenizer(['你', '好', '呀'])
        ids = self.vocab.encode('你好呀') + self.vocab.encode('你好')
        self.stream = DocumentStream(np.array(ids, dtype=np.uint16), BOS, EOS, PAD)
        self.provenance = {'tokens_sha256': 'tokens', 'index_sha256': 'index', 'vocab_sha256': 'vocab',
                           'verification_sha256': 'verification', 'admission_sha256': 'review',
                           'batch': 2, 'length': 8, 'device': 'cpu'}

    def new_model(self):
        return HybridLanguageBridge(self.organism, self.vocab, 'vocab', context_layers=1)

    def test_aligned_loss_matches_manual_and_rejects_shift(self):
        model = self.new_model()
        trainer = PackedHybridTrainer(model, self.state, 'parent', self.provenance, 7)
        inputs = torch.tensor([[BOS, self.vocab.ids['你'], self.vocab.ids['好'], PAD]])
        labels = torch.tensor([[self.vocab.ids['你'], self.vocab.ids['好'], EOS, -100]])
        expected = torch.nn.functional.cross_entropy(model(inputs, self.state).reshape(-1, self.vocab.size),
                                                      labels.reshape(-1), ignore_index=-100).item()
        self.assertAlmostEqual(trainer.step_aligned(inputs, labels), expected, places=6)
        self.assertEqual(trainer.tokens_seen, 3)
        wrong = labels.clone()
        wrong[0, 0] = self.vocab.ids['好']
        with self.assertRaisesRegex(ValueError, 'not next-token aligned'):
            trainer.step_aligned(inputs, wrong)
        with self.assertRaisesRegex(ValueError, 'empty'):
            trainer.step_aligned(inputs, torch.full_like(labels, -100))

    def test_resume_replays_next_batch_and_preserves_individual(self):
        model = self.new_model()
        trainer = PackedHybridTrainer(model, self.state, 'parent', self.provenance, 13)
        before = frozen_digest(self.organism, self.state)
        trainer.step(self.stream)
        checkpoint = copy.deepcopy(trainer.snapshot())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint.pt'
            atomic_save(checkpoint, path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        restored = self.new_model()
        resumed = PackedHybridTrainer(restored, self.state, 'parent', self.provenance, 13)
        resumed.restore(checkpoint)
        self.assertEqual(trainer.step(self.stream), resumed.step(self.stream))
        self.assertEqual(trainer.tokens_seen, resumed.tokens_seen)
        self.assertEqual(trainer.packed_rng.getstate(), resumed.packed_rng.getstate())
        self.assertEqual(trainer.coverage.metrics(), resumed.coverage.metrics())
        self.assertEqual(trainer.coverage.metrics()['target_presentations'], trainer.tokens_seen)
        for name, tensor in interface_state(model).items():
            self.assertTrue(torch.equal(tensor, interface_state(restored)[name]))
        self.assertEqual(before, frozen_digest(self.organism, self.state))
        with self.assertRaisesRegex(ValueError, 'Packed stream'):
            ByteTrainer(self.new_model(), self.state, 'parent', self.provenance, 13,
                        format_name='spora-hybrid-training-v1').restore(checkpoint)
        wrong = dict(self.provenance, tokens_sha256='other')
        with self.assertRaisesRegex(ValueError, 'identical data'):
            PackedHybridTrainer(self.new_model(), self.state, 'parent', wrong, 13).restore(checkpoint)

    def test_coverage_counts_repeated_and_overlapping_windows(self):
        coverage = PackedCoverage(self.stream)
        coverage.add([(0, 0, 2), (0, 1, 3), (0, 0, 2)])
        self.assertEqual(coverage.metrics(), {'target_presentations': 6, 'distinct_target_positions': 3,
                                               'distinct_documents': 1, 'distinct_windows': 2})
        saved = coverage.snapshot()
        restored = PackedCoverage.restore(self.stream, saved)
        self.assertEqual(restored.metrics(), coverage.metrics())
        saved['positions'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'Inconsistent'):
            PackedCoverage.restore(self.stream, saved)
        with self.assertRaisesRegex(ValueError, 'Invalid supervised span'):
            coverage.add([(0, 0, int(self.stream.ends[0]))])


if __name__ == '__main__':
    unittest.main()
