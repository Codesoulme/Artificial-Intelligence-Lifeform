import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hybrid_tokenizer import audit
from spora.hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer, windows
from spora.language_training import ByteTrainer, interface_state, load_interface
from spora.growing import new_state
from spora.temporal import TemporalConfig, TemporalSPORA
from spora.train_hybrid import validate_diagnostic_review
import torch


class HybridTokenizerTests(unittest.TestCase):
    def test_roundtrip_mixed_unicode_and_fallback(self):
        vocab = HybridTokenizer(['你', '好', '，'])
        for text in ('', '你好，世界!', '未知🙂e\u0301', '你\x00好', '你好'):
            self.assertEqual(vocab.decode(vocab.encode(text)), text)
        self.assertEqual(len(vocab.encode('你🙂')) - 2, 5)

    def test_rejects_invalid_content_and_invalid_utf8(self):
        with self.assertRaises(ValueError):
            HybridTokenizer(['你好'])
        with self.assertRaises(ValueError):
            HybridTokenizer(['你', '你'])
        vocab = HybridTokenizer(['你'])
        with self.assertRaises(UnicodeDecodeError):
            vocab.decode([0xe4, vocab.ids['你']])
        with self.assertRaises(ValueError):
            vocab.decode([vocab.size])

    def test_audit_uses_training_vocabulary_without_opening_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for split, text in [('train', '你好'), ('development', '你🙂')]:
                data = text.encode('utf8')
                (root / split).write_bytes(data)
                rows.append({'split': split, 'path': split, 'sha256': hashlib.sha256(data).hexdigest()})
            rows.append({'split': 'test', 'path': 'does-not-exist', 'sha256': 'not-read'})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'schema': 'spora-text-v1', 'documents': rows}), encoding='utf8')
            vocab = root / 'vocab.json'
            vocab.write_text(json.dumps({'format': 'spora-codepoint-vocab-v1', 'symbols': ['你', '好']}), encoding='utf8')
            result = audit(manifest, vocab)
            self.assertEqual(result['splits']['development']['fallback_characters'], 1)
            self.assertEqual(result['splits']['development']['roundtrip_failures'], 0)
            self.assertFalse(result['test_split_read'])
            vocab.write_text(json.dumps({'format': 'spora-codepoint-vocab-v1', 'symbols': ['你', '好', '🙂']}), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'training characters'):
                audit(manifest, vocab)

    def test_hybrid_checkpoint_binding_and_resume(self):
        torch.manual_seed(7)
        organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        vocab = HybridTokenizer(['你', '好'])
        model = HybridLanguageBridge(organism, vocab, 'vocab-hash', context_layers=0)
        trainer = ByteTrainer(model, new_state(organism), 'parent', {'vocab_sha256': 'vocab-hash'}, 42,
                              format_name='spora-hybrid-training-v1')
        data = windows([{'split': 'train', 'text': '你好🙂'}], vocab, 'train', 16)
        trainer.step(data, 1)
        snapshot = trainer.snapshot()
        restored = HybridLanguageBridge(organism, vocab, 'vocab-hash', context_layers=0)
        resumed = ByteTrainer(restored, new_state(organism), 'parent', {'vocab_sha256': 'vocab-hash'}, 42,
                              format_name='spora-hybrid-training-v1')
        resumed.restore(snapshot)
        self.assertEqual(resumed.steps, 1)
        self.assertEqual(resumed.tokens_seen, trainer.tokens_seen)
        for key, tensor in interface_state(model).items():
            self.assertTrue(torch.equal(tensor, interface_state(restored)[key]))
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            load_interface(HybridLanguageBridge(organism, vocab, 'other', context_layers=0), snapshot, 'parent')
        wrong = dict(snapshot, format='spora-codepoint-training-v1')
        with self.assertRaisesRegex(ValueError, 'format mismatch'):
            load_interface(restored, wrong, 'parent')

    def test_noisy_story_review_is_bounded_and_keeps_defect_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            basis = Path(temp) / 'review.json'
            basis.write_text(json.dumps({'approved_for_pilot': True, 'sample_count': 64,
                                         'counts': {'clean': 34, 'minor': 24, 'major': 6}}), encoding='utf8')
            review = {'experiment': 'hybrid-character-byte-diagnostic-v1', 'audit_sha256': 'audit',
                      'policy': 'bounded-noisy-story-v1', 'approved_for_pilot': True,
                      'manifest_sha256': 'manifest', 'parent_checkpoint_sha256': 'parent',
                      'max_schedule_steps': 100, 'source_review_path': str(basis),
                      'source_review_sha256': hashlib.sha256(basis.read_bytes()).hexdigest()}
            args = {'manifest_sha': 'manifest', 'parent_sha': 'parent', 'audit_sha': 'audit', 'steps': 100}
            result = validate_diagnostic_review(review, **args)
            self.assertEqual(result['source_review_counts']['major'], 6)
            self.assertFalse(result['independent_human_review'])
            with self.assertRaises(ValueError):
                validate_diagnostic_review(review, **dict(args, steps=101))
            with self.assertRaises(ValueError):
                validate_diagnostic_review(review, **dict(args, manifest_sha='other'))


if __name__ == '__main__':
    unittest.main()
