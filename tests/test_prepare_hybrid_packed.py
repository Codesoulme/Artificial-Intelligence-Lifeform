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


if __name__ == '__main__':
    unittest.main()
