import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.audit_story_transfer import audit
from spora.hybrid_tokenizer import HybridTokenizer


class StoryTransferAuditTests(unittest.TestCase):
    def test_expansion_preserves_old_data_and_development(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def make_manifest(folder, train, dev):
                folder.mkdir()
                rows = []
                for split, texts in [('train', train), ('development', dev)]:
                    for index, text in enumerate(texts):
                        data = text.encode('utf8')
                        name = f'{split}-{index}'
                        (folder / name).write_bytes(data)
                        rows.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(),
                                     'split': split, 'source': 'fixture', 'license': 'fixture'})
                path = folder / 'manifest.json'
                path.write_text(json.dumps({'schema': 'spora-text-v1', 'documents': rows}), encoding='utf8')
                return path

            old = make_manifest(root / 'old', ['甲乙'], ['甲丙'])
            new = make_manifest(root / 'new', ['甲乙', '丁戊'], ['甲丙'])
            wrong = make_manifest(root / 'wrong', ['甲乙', '丁戊'], ['甲己'])
            tokenizer = root / 'tokenizer.json'
            tokenizer.write_text(json.dumps(HybridTokenizer(['乙', '甲']).payload(), ensure_ascii=False), encoding='utf8')
            checkpoint = root / 'checkpoint.pt'
            torch.save({'format': 'spora-hybrid-training-v1', 'provenance': {
                'manifest_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
                'vocab_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest()}}, checkpoint)
            result = audit(old, new, tokenizer, checkpoint)
            self.assertEqual(result['additional_train_documents'], 1)
            self.assertEqual(result['splits']['train']['fallback_characters'], 2)
            self.assertEqual(result['splits']['train']['roundtrip_failures'], 0)
            with self.assertRaisesRegex(ValueError, 'unchanged development'):
                audit(old, wrong, tokenizer, checkpoint)

    def test_transfer_accepts_existing_vocabulary_with_byte_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            def manifest(folder, train):
                folder.mkdir()
                rows = []
                for split, texts in [('train', train), ('development', ['甲乙'])]:
                    for index, text in enumerate(texts):
                        raw = text.encode('utf8')
                        name = f'{split}-{index}'
                        (folder / name).write_bytes(raw)
                        rows.append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest(),
                                     'split': split, 'source': 'fixture', 'license': 'fixture'})
                path = folder / 'manifest.json'
                path.write_text(json.dumps({'schema': 'spora-text-v1', 'documents': rows}), encoding='utf8')
                return path

            old = manifest(root / 'old', ['甲丙'])
            new = manifest(root / 'new', ['甲丙', '戊己'])
            tokenizer = root / 'tokenizer.json'
            tokenizer.write_text(json.dumps(HybridTokenizer(['甲']).payload(), ensure_ascii=False), encoding='utf8')
            checkpoint = root / 'checkpoint.pt'
            torch.save({'format': 'spora-hybrid-training-v1', 'provenance': {
                'manifest_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
                'vocab_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest()}}, checkpoint)
            result = audit(old, new, tokenizer, checkpoint)
            self.assertEqual(result['additional_train_documents'], 1)
            self.assertEqual(result['splits']['train']['fallback_characters'], 3)


if __name__ == '__main__':
    unittest.main()
