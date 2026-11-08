import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from scripts import p115_encode_stories as encode


class EncodeStoriesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = self.root/'runs'
        self.runs.mkdir()
        self.source = self.root/'input.jsonl'
        self.token_path = self.root/'tokenizer.json'
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        vocab = {c: i for i, c in enumerate(sorted(alphabet))}
        vocab.update({'<|bos|>': 256, '<|eos|>': 257})
        tokenizer = Tokenizer(models.BPE(vocab, []))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.save(str(self.token_path))
        self.tokenizer = tokenizer
        self.patcher = patch.object(encode, 'RUNS', self.runs)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def row(self, text, n):
        full, group = encode.identities(text)
        return dict(text=text, sha256=full, group=group,
                    source_original_sha256=encode.digest(text), source_row_1based=n, origin='synthetic')

    def write(self, rows):
        self.source.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')

    def run_it(self, name='result'):
        return encode.run(self.source, self.runs/name, True, self.token_path)

    def test_binary_offsets_rejections_and_provenance(self):
        texts = ['hello', '好!', 'x'*510, 'z'*511, '']
        self.write([self.row(t, i+1) for i, t in enumerate(texts)])
        result = self.run_it()
        out = self.runs/'result'
        rows = [json.loads(s) for s in (out/'stories.jsonl').read_text(encoding='utf-8').splitlines()]
        data = (out/'stories.bin').read_bytes()
        expected_offset = 0
        for row, text in zip(rows, texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            expected = [256, *ids, 257]
            self.assertEqual(row['token_offset'], expected_offset)
            self.assertEqual(row['token_count'], len(expected))
            self.assertEqual(data[2*expected_offset:2*(expected_offset+len(expected))], struct.pack('<'+'H'*len(expected), *expected))
            self.assertEqual(row['input_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
            self.assertEqual(row['origin'], 'synthetic')
            expected_offset += len(expected)
        self.assertEqual(len(data), 2*expected_offset)
        self.assertEqual(result['status'], 'candidate_not_trained')
        self.assertEqual(result['counts']['within_document_next_token_targets'], expected_offset-len(rows))
        self.assertEqual(result['counts']['rejection_reasons'], {'over_510_content_tokens': 1, 'empty_content': 1})
        self.assertEqual(len((out/'rejected.jsonl').read_text().splitlines()), 2)
        self.assertEqual(result['counts']['unique_groups'], 5)
        self.assertEqual(result['content_token_length_histogram']['510'], 1)

    def test_reject_roundtrip(self):
        self.tokenizer.normalizer = normalizers.Lowercase()
        self.tokenizer.save(str(self.token_path))
        self.write([self.row('UPPER', 1)])
        result = self.run_it()
        self.assertEqual(result['counts']['rejection_reasons'], {'tokenizer_roundtrip': 1})
        self.assertEqual((self.runs/'result'/'stories.bin').read_bytes(), b'')

    def test_invalid_identity_and_duplicates_fail_entire_run(self):
        good = self.row('One', 1)
        variants = [dict(good, sha256='bad'), dict(good, group='bad'),
                    dict(good, source_original_sha256='bad'), dict(good, source_row_1based=True)]
        cases = [[v] for v in variants] + [[good, self.row('two', 1)], [good, self.row(' ONE ', 2)]]
        for i, rows in enumerate(cases):
            with self.subTest(i=i):
                self.write(rows)
                with self.assertRaises(ValueError):
                    self.run_it(str(i))
                out = self.runs/str(i)
                self.assertFalse((out/'summary.json').exists())
                self.assertEqual(json.loads((out/'failure.json').read_text())['status'], 'failed_do_not_use_partial_outputs')

    def test_no_overwrite_or_unconfirmed_input(self):
        self.write([self.row('abc', 1)])
        self.run_it()
        before = (self.runs/'result'/'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_it()
        self.assertEqual(before, (self.runs/'result'/'summary.json').read_bytes())
        with self.assertRaises(ValueError):
            encode.run(self.source, self.runs/'other', False, self.token_path)
        self.assertFalse((self.runs/'other').exists())

    def test_input_change_is_detected(self):
        self.write([self.row('abc', 1)])
        real_hash = encode.file_hash
        calls = 0
        def changing_hash(path):
            nonlocal calls
            if path == self.source:
                calls += 1
                if calls == 2:
                    self.write([self.row('changed', 2)])
            return real_hash(path)
        with patch.object(encode, 'file_hash', side_effect=changing_hash):
            with self.assertRaisesRegex(ValueError, 'changed during conversion'):
                self.run_it()
        self.assertFalse((self.runs/'result'/'summary.json').exists())
        failure = json.loads((self.runs/'result'/'failure.json').read_text())
        self.assertEqual(failure['stage'], 'input_verification')

    def test_existing_tokenizer_with_synthetic_text(self):
        self.write([self.row('Hello, world!', 1)])
        result = encode.run(self.source, self.runs/'existing', True, encode.TOKENIZER)
        self.assertEqual(result['counts']['encoded_documents'], 1)
        record = json.loads((self.runs/'existing'/'stories.jsonl').read_text(encoding='utf-8'))
        binary = (self.runs/'existing'/'stories.bin').read_bytes()
        ids = struct.unpack('<'+'H'*record['token_count'], binary)
        tok = Tokenizer.from_file(str(encode.TOKENIZER))
        self.assertEqual(ids[0], tok.token_to_id('<|bos|>'))
        self.assertEqual(ids[-1], tok.token_to_id('<|eos|>'))
        self.assertEqual(tok.decode(list(ids[1:-1]), skip_special_tokens=False), 'Hello, world!')


if __name__ == '__main__':
    unittest.main()
