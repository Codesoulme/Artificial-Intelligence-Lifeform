import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import unicodedata
import re
import subprocess
import sys
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p114', ROOT/'scripts/p114_disk_story_dedupe.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def row(text):
    normalized = re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
    return dict(text=text, group='synthetic', sha256=p.digest(normalized), source_original_sha256=p.digest(text))


class DiskDedupeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'runs', prefix='p114-test-')
        self.base = Path(self.temp.name)
        self.source = self.base/'input.jsonl'

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        self.source.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf-8')

    def run_case(self, name, **kw):
        out = self.base/name
        summary = p.run(self.source, out, **kw)
        groups = [json.loads(x) for x in (out/'groups.jsonl').read_text(encoding='utf-8').splitlines()]
        return out, summary, groups

    def test_deep_path_compression_and_transaction_rollback(self):
        db = sqlite3.connect(':memory:')
        self.addCleanup(db.close)
        db.execute('CREATE TABLE records(n INTEGER PRIMARY KEY, parent INTEGER)')
        db.executemany('INSERT INTO records VALUES (?,?)', ((i, max(1, i-1)) for i in range(1, 301)))
        db.commit()
        before = list(db.execute('SELECT n,parent FROM records ORDER BY n'))
        self.assertEqual(p.root(db, 300), 1)
        self.assertEqual(db.execute('SELECT COUNT(*) FROM records WHERE parent != 1').fetchone()[0], 0)
        db.rollback()
        self.assertEqual(list(db.execute('SELECT n,parent FROM records ORDER BY n')), before)
        p.union(db, 300, 1)
        db.commit()
        self.assertEqual(db.execute('SELECT COUNT(*) FROM records WHERE parent != 1').fetchone()[0], 0)
        db.execute('INSERT INTO records VALUES(301,301)')
        p.union(db, 301, 200)
        self.assertEqual(p.root(db, 301), 1)

    def test_compression_preserves_original_outputs_and_risks(self):
        prefix = ''.join(chr(0x4e00+i) for i in range(96))
        self.write([row(prefix+'甲'), row(prefix+'乙'), row('天地玄黄宇宙洪荒'), row('短')])
        def old_root(db, n):
            while True:
                parent = db.execute('SELECT parent FROM records WHERE n=?', (n,)).fetchone()[0]
                if parent == n:
                    return n
                n = parent
        with patch.object(p, 'signature', return_value=(1,)*32):
            new, _, groups = self.run_case('compressed', max_bucket=2)
            with patch.object(p, 'root', side_effect=old_root):
                old, _, _ = self.run_case('uncompressed', max_bucket=2)
        self.assertTrue(all(x['status'] == 'quarantine' for x in groups))
        self.assertEqual([x['representative_line'] for x in groups], [1, 1, 3, 4])
        for filename in ('candidates.jsonl', 'quarantine.jsonl', 'groups.jsonl', 'unchecked_buckets.jsonl'):
            self.assertEqual((new/filename).read_bytes(), (old/filename).read_bytes())

    def test_command_line_runs_with_documented_arguments(self):
        self.write([row('天地玄黄宇宙洪荒')])
        out = self.base/'cli'
        result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/p114_disk_story_dedupe.py'),
            '--input', str(self.source), '--output', str(out), '--min-available-mib', '0'],
            capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((out/'summary.json').read_text(encoding='utf-8'))['counts']['candidates'], 1)

    def test_benchmark_counts_known_pairs_and_rejects_incomplete_output(self):
        spec = importlib.util.spec_from_file_location('benchmark', ROOT/'scripts/p114_synthetic_benchmark.py')
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        data = [dict(source_line=i, representative_line=i, status='candidate') for i in range(1, 13)]
        data[1]['representative_line'] = 1
        data[11]['status'] = 'quarantine'
        path = self.base/'groups.jsonl'
        path.write_text(''.join(json.dumps(x)+'\n' for x in data), encoding='utf-8')
        self.assertEqual(benchmark.constructed_pair_counts(path, 12),
                         {'expected': 2, 'same_group': 1, 'quarantined': 1})
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            benchmark.constructed_pair_counts(path, 13)

    def test_cross_batch_prefix_bridge_coverage_and_determinism(self):
        prefix = ''.join(chr(0x4e00+i) for i in range(96))
        a = prefix + ''.join(chr(0x5100+i) for i in range(100))
        b = prefix + ''.join(chr(0x5200+i) for i in range(100))
        c = '甲'+b[1:]
        fillers = [row(''.join(chr(0x6000+(i*13+j)%2000) for j in range(25))) for i in range(101)]
        self.write([row(a), row(b)] + fillers + [row(c), row(a)])
        with patch.object(p, 'signature', return_value=(1,)*32):
            one, summary, groups = self.run_case('one', max_bucket=200, max_pairs=1000000)
            two, _, _ = self.run_case('two', max_bucket=200, max_pairs=1000000)
        self.assertEqual([x['source_line'] for x in groups], list(range(1, 106)))
        self.assertEqual([groups[i-1]['representative_line'] for i in (1,2,104,105)], [1]*4)
        self.assertGreater(summary['counts']['similar_pairs'], 0)
        for filename in ('candidates.jsonl', 'quarantine.jsonl', 'groups.jsonl', 'unchecked_buckets.jsonl'):
            self.assertEqual((one/filename).read_bytes(), (two/filename).read_bytes())

    def test_skipped_bucket_propagates_to_prefix_component(self):
        prefix = ''.join(chr(0x4e00+i) for i in range(96))
        self.write([row(prefix+'甲'), row(prefix+'乙'), row('天地玄黄宇宙洪荒')])
        with patch.object(p, 'signature', return_value=(1,)*32):
            out, summary, groups = self.run_case('big', max_bucket=2)
        self.assertEqual(summary['counts']['quarantined_records'], 3)
        self.assertTrue(all('oversized_bucket' in x['review_reasons'] for x in groups))
        self.assertEqual(len((out/'unchecked_buckets.jsonl').read_text().splitlines()), 8)
        with patch.object(p, 'signature', return_value=(1,)*32):
            _, _, groups = self.run_case('budget', max_pairs=0)
        self.assertTrue(all('pair_budget' in x['review_reasons'] for x in groups))

    def test_hash_validation_short_text_and_protection(self):
        data = row(' Ａ 天地玄黄宇宙')
        self.write([data, row('短')])
        out, _, groups = self.run_case('valid')
        self.assertEqual(groups[1]['status'], 'quarantine')
        old = (out/'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            p.run(self.source, out)
        self.assertEqual(old, (out/'summary.json').read_bytes())
        for key in ('sha256', 'source_original_sha256'):
            bad = dict(data); bad[key] = 'bad'; self.write([bad])
            with self.assertRaisesRegex(ValueError, 'sha256'):
                p.run(self.source, self.base/('bad-'+key))
            self.assertTrue((self.base/('bad-'+key)/'failure.json').exists())
        self.write([row('天'*1000)])
        with self.assertRaisesRegex(ValueError, 'max_line_bytes'):
            p.run(self.source, self.base/'limit', max_line_bytes=256)
        self.assertEqual(json.loads((self.base/'limit'/'failure.json').read_text())['stage'], 'ingest')

    def test_resource_guard_and_full_grams(self):
        self.write([row('天地玄黄宇宙洪荒'), row('天地玄黄宇宙洪荒甲')])
        m = dict(available_physical=10**12, working_set=0, peak_working_set=0, commit=10**12, peak_commit=10**12)
        with patch.object(p, 'memory_info', return_value=m), self.assertRaisesRegex(RuntimeError, 'memory guard'):
            p.run(self.source, self.base/'mem')
        self.assertTrue((self.base/'mem'/'failure.json').exists())
        with patch.object(p.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()), self.assertRaisesRegex(RuntimeError, 'disk guard'):
            p.run(self.source, self.base/'disk')
        self.assertTrue((self.base/'disk'/'failure.json').exists())
        with patch.object(p, 'signature', return_value=(1,)*32):
            _, summary, groups = self.run_case('verify', threshold=.81)
        self.assertEqual(summary['counts']['candidate_pairs_verified'], 1)
        self.assertEqual(summary['counts']['similar_pairs'], 0)
        self.assertTrue(all(x['status'] == 'candidate' for x in groups))
        with patch.object(p, 'signature', return_value=(1,)*32):
            _, summary, _ = self.run_case('inclusive', threshold=.8)
        self.assertEqual(summary['counts']['similar_pairs'], 1)


if __name__ == '__main__':
    unittest.main()
