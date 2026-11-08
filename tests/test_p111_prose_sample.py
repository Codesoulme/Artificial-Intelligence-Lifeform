import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.p111_prose_sample import quotas, run, signals
from scripts.p110_data_audit import signals as original_signals


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'input.jsonl'
        self.good = '山间清风吹过树林，远处云影缓缓移动。' * 8

    def write(self, rows):
        data = b''.join((json.dumps(row, ensure_ascii=False) + '\n').encode() for row in rows)
        self.path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def row(self, group, bucket, text=None):
        return {'text': self.good if text is None else text, 'group': group,
                'source_byte_bucket': bucket, 'extra': 'preserved'}

    def test_rebuild_groups_and_quarantine_across_buckets(self):
        rows = [self.row('a', 0), self.row('a', 1), self.row('b', 1),
                self.row('c', 0), self.row('c', 1, self.good + ' test@example.com'),
                self.row('d', 0), self.row('e', 1, self.good + '\ufffd')]
        expected = self.write(rows)
        a = run(self.path, self.root / 'a', expected, 3, 42)
        b = run(self.path, self.root / 'b', expected, 3, 42)
        self.assertEqual(a, b)
        self.assertEqual((self.root / 'a/sample.jsonl').read_bytes(), (self.root / 'b/sample.jsonl').read_bytes())
        sample = [json.loads(x) for x in (self.root / 'a/sample.jsonl').read_bytes().splitlines()]
        self.assertEqual({r['group'] for r in sample}, {'a', 'b', 'd'})
        self.assertEqual(len(sample), len({r['group'] for r in sample}))
        self.assertTrue(all(r in rows for r in sample))
        self.assertEqual(a['quarantined_groups'], 2)
        self.assertEqual(a['buckets']['0']['sampled_groups'], 2)

    def test_bad_hash_bad_record_and_insufficient_groups_leave_no_output(self):
        expected = self.write([self.row('a', 0)])
        for name, digest, size in [('hash', '0' * 64, 1), ('short', expected, 2)]:
            with self.assertRaises(ValueError):
                run(self.path, self.root / name, digest, size)
            self.assertFalse((self.root / name).exists())
        expected = self.write([{'text': 'bad'}])
        with self.assertRaises(ValueError):
            run(self.path, self.root / 'invalid', expected, 1)
        self.assertFalse((self.root / 'invalid').exists())

    def test_existing_output_is_untouched(self):
        expected = self.write([self.row('a', 0)])
        target = self.root / 'out'
        target.mkdir()
        sentinel = target / 'sample.jsonl'
        sentinel.write_text('unchanged')
        with self.assertRaises(FileExistsError):
            run(self.path, target, expected, 1)
        self.assertEqual(sentinel.read_text(), 'unchanged')

    def test_predicates_match_p110(self):
        for text in [self.good, self.good + ' test@example.com', self.good + '\ufffd',
                     self.good + '13812345678', '问：哪里？', self.good + 'a' * 12]:
            old = original_signals(text)
            self.assertEqual(signals(text), {k: old[k] for k in signals(text)})

    def test_largest_remainder_ties_by_bucket(self):
        self.assertEqual(quotas({0: 1, 1: 1, 2: 1}, 2), {0: 1, 1: 1, 2: 0})


if __name__ == '__main__':
    unittest.main()
