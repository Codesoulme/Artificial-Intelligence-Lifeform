import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hybrid_tokenizer import audit
from spora.hybrid_tokenizer import HybridTokenizer


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


if __name__ == '__main__':
    unittest.main()
