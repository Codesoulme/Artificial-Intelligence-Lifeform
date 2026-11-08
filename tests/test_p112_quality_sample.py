import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts import p112_quality_sample as sample


def row(text, index, group=None):
    full, computed_group = sample.identities(text)
    return {'full': full, 'group': group or computed_group, 'index': index,
            'original': sample.sha_text(text), 'content_tokens': 3}


class QualitySampleTests(unittest.TestCase):
    def test_selection_excludes_original_full_and_group_and_is_order_independent(self):
        originals = [row('original one', 1), row('original two', 2)]
        candidates = [dict(originals[0]), row('shared group', 3, originals[1]['group']),
                      row('eligible one', 4), row('eligible two', 5), row('eligible three', 6)]
        a, b, count = sample.choose(originals, candidates, 2)
        self.assertEqual(count, 3)
        self.assertEqual((a, b, count), sample.choose(originals[::-1], candidates[::-1], 2))
        self.assertTrue(all(r['index'] >= 4 for r in b))
        self.assertEqual([r['full'] for r in b], [r['full'] for r in sorted(candidates[2:], key=lambda x: (sample.score(x['full']), x['full']))[:2]])

    def test_database_reads_only_train_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.sqlite'
            db = sqlite3.connect(path)
            try:
                db.execute('CREATE TABLE candidates (full_hash, group_hash, row_index, original_sha, content_tokens, split, held_text)')
                db.execute("INSERT INTO candidates VALUES ('f', 'g', 1, 'o', 3, 'train', NULL)")
                db.execute("INSERT INTO candidates VALUES ('held', 'g', 2, 'o', 3, 'test', 'SECRET')")
                db.commit()
            finally:
                db.close()
            rows, digest = sample.read_candidates(path)
            self.assertEqual(rows, [{'full': 'f', 'group': 'g', 'index': 1, 'original': 'o', 'content_tokens': 3}])
            self.assertEqual(len(digest), 64)

    def test_parquet_selected_rows_and_identity_failure(self):
        class Scalar:
            def __init__(self, value): self.value = value
            def as_py(self): return self.value
        class Table:
            def __init__(self, values): self.values = values
            def column(self, index): return [Scalar(v) for v in self.values]
        class Metadata:
            def row_group(self, index): return type('Group', (), {'num_rows': 2})()
        class Parquet:
            num_row_groups = 2
            metadata = Metadata()
            def __init__(self): self.reads = []
            def read_row_group(self, index, columns, use_threads):
                self.reads.append((index, columns, use_threads))
                return Table([['a', 'b'], ['c', 'd']][index])
        parquet = Parquet()
        chosen = [row('d', 3)]
        self.assertEqual(sample.extract(parquet, chosen), {3: 'd'})
        self.assertEqual(parquet.reads, [(1, ['story'], False)])
        chosen[0]['original'] = 'wrong'
        with self.assertRaises(ValueError): sample.extract(parquet, chosen)

    def test_duplicate_or_short_pool_fails(self):
        r = row('a', 0)
        with self.assertRaises(ValueError): sample.choose([r, r], [], 1)
        with self.assertRaises(ValueError): sample.choose([r], [], 1)


if __name__ == '__main__':
    unittest.main()
