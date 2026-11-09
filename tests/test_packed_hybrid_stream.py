import hashlib
import gc
import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_hybrid_packed import prepare
from scripts.verify_hybrid_packed import verify
from spora.hybrid_tokenizer import HybridTokenizer
from spora.language_bridge import BOS, EOS, PAD
from spora.packed_hybrid_stream import open_packed_hybrid


class PackedHybridStreamTests(unittest.TestCase):
    def test_sampler_never_crosses_documents_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vocab = HybridTokenizer(['你', '好'])
            vocab_path = root / 'vocab.json'
            vocab_path.write_text(json.dumps(vocab.payload(), ensure_ascii=False), encoding='utf8')
            records = []
            for index, text in enumerate(['你好', '你好🙂']):
                records.append({'split': 'train', 'text': text, 'source_row_index': index, 'group': str(index),
                                'source_original_sha256': hashlib.sha256(text.encode('utf8')).hexdigest()})
            source = root / 'train.jsonl'
            raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode('utf8')
            source.write_bytes(raw)
            directory = root / 'packed'
            prepare(source, vocab_path, directory, expected_source_sha=hashlib.sha256(raw).hexdigest(), expected_rows=2)
            verification = root / 'verification.json'
            verification.write_text(json.dumps(verify(directory, vocab_path, source)), encoding='utf8')
            stream = open_packed_hybrid(directory, vocab_path, verification)
            inputs, labels = stream.batch(random.Random(11), 100, 8, 'cpu')
            for x, y in zip(inputs.tolist(), labels.tolist()):
                active = [token for token in y if token != -100]
                self.assertNotIn(BOS, active)
                if EOS in active:
                    self.assertEqual(active[-1], EOS)
                self.assertNotIn(PAD, active)
            with (directory / 'index.jsonl').open('ab') as changed:
                changed.write(b'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                open_packed_hybrid(directory, vocab_path, verification)
            del stream
            gc.collect()


if __name__ == '__main__':
    unittest.main()
