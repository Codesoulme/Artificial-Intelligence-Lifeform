import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hybrid_tokenizer import audit
from spora.hybrid_tokenizer import HybridLanguageBridge, HybridTokenizer, windows
from spora.language_training import ByteTrainer, frozen_digest, interface_state, load_interface
from spora.growing import new_state
from spora.temporal import TemporalConfig, TemporalSPORA
from spora.train_hybrid import rebind_resume_review, validate_diagnostic_review
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
        original_loss = trainer.step(data, 1)
        resumed_loss = resumed.step(data, 1)
        self.assertEqual(original_loss, resumed_loss)
        for key, tensor in interface_state(model).items():
            self.assertTrue(torch.equal(tensor, interface_state(restored)[key]))
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            load_interface(HybridLanguageBridge(organism, vocab, 'other', context_layers=0), snapshot, 'parent')
        wrong = dict(snapshot, format='spora-codepoint-training-v1')
        with self.assertRaisesRegex(ValueError, 'format mismatch'):
            load_interface(restored, wrong, 'parent')

    def test_copied_blocks_start_identical_and_leave_organism_frozen(self):
        torch.manual_seed(17)
        organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=16, expert_rank=4, memory_layers=(0, 1), history_writer='rls'))
        state = new_state(organism)
        vocab = HybridTokenizer(['你', '好'])
        model = HybridLanguageBridge(organism, vocab, 'vocab-hash', context_layers=1)
        tokens = torch.tensor([vocab.encode('你好')])
        baseline = model(tokens, state).detach()
        before = frozen_digest(organism, state)
        model.enable_language_block_copies()
        self.assertTrue(torch.equal(baseline, model(tokens, state).detach()))
        trainer = ByteTrainer(model, state, 'parent', {'vocab_sha256': 'vocab-hash'}, 2,
                              format_name='spora-hybrid-training-v1')
        trainer.step_batch(tokens)
        self.assertEqual(before, frozen_digest(organism, state))
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.adapted_blocks.parameters()))
        restored = HybridLanguageBridge(organism, vocab, 'vocab-hash', context_layers=1)
        with self.assertRaisesRegex(ValueError, 'adapter tensors'):
            load_interface(restored, trainer.snapshot(), 'parent')
        restored.enable_language_block_copies()
        load_interface(restored, trainer.snapshot(), 'parent')
        resumed = ByteTrainer(restored, new_state(organism), 'parent', {'vocab_sha256': 'vocab-hash'}, 2,
                              format_name='spora-hybrid-training-v1')
        resumed.restore(trainer.snapshot())
        self.assertEqual(trainer.step_batch(tokens), resumed.step_batch(tokens))
        for key, tensor in interface_state(model).items():
            self.assertTrue(torch.equal(tensor, interface_state(restored)[key]))

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

    def test_resume_review_extension_requires_identical_training_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            basis = root / 'basis.json'
            basis.write_text(json.dumps({'approved_for_pilot': True, 'sample_count': 64,
                                         'counts': {'major': 6}}), encoding='utf8')
            old = {'experiment': 'hybrid-character-byte-diagnostic-v1', 'audit_sha256': 'audit',
                   'policy': 'bounded-noisy-story-v1', 'approved_for_pilot': True,
                   'manifest_sha256': 'manifest', 'parent_checkpoint_sha256': 'parent',
                   'max_schedule_steps': 100, 'source_review_path': str(basis),
                   'source_review_sha256': hashlib.sha256(basis.read_bytes()).hexdigest()}
            old_path = root / 'old.json'
            old_path.write_text(json.dumps(old), encoding='utf8')
            old_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
            new_review = dict(old, previous_review_path=str(old_path), previous_review_sha256=old_sha,
                              max_schedule_steps=500)
            payload = {'steps': 100, 'provenance': {'review_sha256': old_sha, 'batch': 4}}
            new_provenance = {'review_sha256': 'new', 'batch': 4}
            args = {'manifest_sha': 'manifest', 'parent_sha': 'parent', 'audit_sha': 'audit'}
            rebound = rebind_resume_review(payload, new_provenance, new_review, **args)
            self.assertEqual(rebound['provenance'], new_provenance)
            self.assertEqual(payload['provenance']['review_sha256'], old_sha)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                rebind_resume_review(payload, dict(new_provenance, batch=5), new_review, **args)
            with self.assertRaisesRegex(ValueError, 'Previous review mismatch'):
                rebind_resume_review(payload, new_provenance, dict(new_review, previous_review_sha256='wrong'), **args)


if __name__ == '__main__':
    unittest.main()
