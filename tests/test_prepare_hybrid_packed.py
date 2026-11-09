import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.prepare_hybrid_packed import prepare
from scripts.verify_hybrid_packed import verify
from spora.hybrid_tokenizer import HybridTokenizer
from spora.language_bridge import BOS, EOS


class PackedHybridTests(unittest.TestCase):
    def test_packed_boundaries_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = HybridTokenizer(['你', '好'])
            (root / 'vocab.json').write_text(json.dumps(tokenizer.payload(), ensure_ascii=False), encoding='utf8')
            rows = []
            for index, text in enumerate(['你好', '你好🙂']):
                rows.append({'split': 'train', 'text': text, 'source_row_index': index, 'group': str(index),
                             'source_original_sha256': hashlib.sha256(text.encode('utf8')).hexdigest()})
            raw = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf8')
            (root / 'train.jsonl').write_bytes(raw)
            args = {'expected_source_sha': hashlib.sha256(raw).hexdigest(), 'expected_rows': 2}
            result = prepare(root / 'train.jsonl', root / 'vocab.json', root / 'out', **args)
            tokens = np.fromfile(root / 'out/train.u16', dtype='<u2').tolist()
            index = [json.loads(line) for line in (root / 'out/index.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertEqual(result['documents'], 2)
            self.assertEqual(result['fallback_characters'], 1)
            for item, row in zip(index, rows):
                piece = tokens[item['offset']:item['offset'] + item['tokens']]
                self.assertEqual((piece[0], piece[-1]), (BOS, EOS))
                self.assertEqual(tokenizer.decode(piece), row['text'])
            verified = verify(root / 'out', root / 'vocab.json', root / 'train.jsonl')
            self.assertEqual(verified['documents'], 2)
            self.assertEqual(verified['fallback_byte_tokens'], 4)
            with self.assertRaises(FileExistsError):
                prepare(root / 'train.jsonl', root / 'vocab.json', root / 'out', **args)
            with (root / 'out/train.u16').open('ab') as stream:
                stream.write(b'xx')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify(root / 'out', root / 'vocab.json', root / 'train.jsonl')

    def test_verifier_rejects_noncanonical_equivalent_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = HybridTokenizer(['你'])
            vocab = root / 'vocab.json'
            vocab.write_text(json.dumps(tokenizer.payload(), ensure_ascii=False), encoding='utf8')
            text = '你'
            row = {'split': 'train', 'text': text, 'source_row_index': 0, 'group': 'g',
                   'source_original_sha256': hashlib.sha256(text.encode('utf8')).hexdigest()}
            source = root / 'train.jsonl'
            raw = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf8')
            source.write_bytes(raw)
            directory = root / 'out'
            prepare(source, vocab, directory, expected_source_sha=hashlib.sha256(raw).hexdigest(), expected_rows=1)
            alternate = np.asarray([BOS, *text.encode('utf8'), EOS], dtype='<u2').tobytes()
            (directory / 'train.u16').write_bytes(alternate)
            index = json.loads((directory / 'index.jsonl').read_text(encoding='utf8'))
            index['tokens'] = 5
            (directory / 'index.jsonl').write_text(json.dumps(index) + '\n', encoding='utf8')
            manifest_path = directory / 'manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf8'))
            manifest['tokens_including_boundaries'] = 5
            manifest['tokens_sha256'] = hashlib.sha256(alternate).hexdigest()
            manifest['index_sha256'] = hashlib.sha256((directory / 'index.jsonl').read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'Source and packed document differ'):
                verify(directory, vocab, source)


if __name__ == '__main__':
    unittest.main()
