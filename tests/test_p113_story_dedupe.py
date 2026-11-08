import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p113', ROOT / 'scripts' / 'p113_story_dedupe.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def row(text):
    return {'text': text, 'group': 'synthetic', 'sha256': hashlib.sha256(text.encode()).hexdigest()}


class StoryDedupeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='p113-test-', dir=ROOT / 'runs')
        self.base = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def process(self, texts, **kw):
        source = self.base / 'source.jsonl'
        source.write_text(''.join(json.dumps(row(t), ensure_ascii=False) + '\n' for t in texts), encoding='utf-8')
        out = self.base / 'result'
        summary = p.run(source, out, **kw)
        groups = json.loads((out / 'groups.json').read_text(encoding='utf-8'))
        return out, summary, groups
    def test_transitive_edges_and_nonrepresentative_prefix_bridge(self):
        # Same first 96 normalized characters joins first two; only second and third overlap.
        prefix = ''.join(chr(0x4e00+i) for i in range(96))
        b = prefix + ''.join(chr(0x5000+i) for i in range(100))
        c = '甲' + b[1:]
        a = prefix + ''.join(chr(0x5200+i) for i in range(100))
        with patch.object(p, 'signature', return_value=(1,)*32):
            out, summary, groups = self.process([a, b, c], threshold=.8, min_df_cap=1000)
        self.assertEqual([g['member_lines'] for g in groups], [[1, 2, 3]])
        self.assertEqual(summary['counts']['similar_pairs'], 1)
        self.assertEqual(len(out.joinpath('candidates.jsonl').read_text(encoding='utf-8').splitlines()), 1)
    def test_exact_threshold_and_short_input(self):
        a = set('abcd')
        b = set('abce')
        self.assertEqual(p.jaccard(a, b), 0.6)
        self.assertEqual(p.chinese_grams('一二三四'), set())
        with patch.object(p, 'signature', return_value=(1,)*32):
            _, _, groups = self.process(['一二三四五六', '一二三四五七', '短'], threshold=1/3, min_df_cap=1000)
        self.assertEqual(groups[0]['member_lines'], [1, 2])
        self.assertEqual(groups[1]['status'], 'quarantine')
    def test_oversized_bucket_quarantines_all_and_reports(self):
        texts = ['天地玄黄宇宙洪荒' + chr(0x4e00+i) for i in range(5)]
        with patch.object(p, 'signature', return_value=(1,)*32):
            out, summary, groups = self.process(texts, max_bucket=2, min_df_cap=1000)
        self.assertEqual(summary['output']['quarantined_records'], 5)
        self.assertEqual(summary['counts']['unchecked_buckets'], 8)
        self.assertEqual(json.loads((out/'unchecked_buckets.json').read_text(encoding='utf-8'))[0]['source_lines'], [1,2,3,4,5])
        self.assertTrue(all('oversized_bucket' in g['review_reasons'] for g in groups))
    def test_pair_budget_and_frequent_features(self):
        texts = ['天地玄黄宇宙洪荒' + chr(0x4e00+i) for i in range(5)]
        with patch.object(p, 'signature', return_value=(1,)*32):
            _, summary, groups = self.process(texts, max_pairs=1, min_df_cap=1000)
        self.assertEqual(summary['output']['quarantined_records'], 5)
        self.assertTrue(all('pair_budget' in g['review_reasons'] for g in groups))
    def test_frequent_feature_is_quarantined(self):
        _, summary, groups = self.process(['天地玄黄宇宙洪荒甲', '天地玄黄宇宙洪荒乙'], max_df=.1, min_df_cap=1)
        self.assertEqual(summary['output']['quarantined_records'], 2)
        self.assertTrue(all('frequent_features_removed' in g['review_reasons'] for g in groups))
    def test_exact_and_prefix_and_no_overwrite(self):
        prefix = '天地玄黄' * 24
        source = self.base / 'source.jsonl'
        texts = [prefix+'甲', prefix+'乙', prefix+'甲']
        source.write_text(''.join(json.dumps(row(t), ensure_ascii=False)+'\n' for t in texts), encoding='utf-8')
        out = self.base / 'result'
        summary = p.run(source, out, min_df_cap=1000)
        self.assertEqual(summary['output']['groups'], 1)
        original = (out/'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            p.run(source, out)
        self.assertEqual(original, (out/'summary.json').read_bytes())
    def test_invalid_hash_rejected(self):
        source = self.base/'bad.jsonl'
        source.write_text(json.dumps({'text':'天地玄黄宇宙','group':'x','sha256':'bad'})+'\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'sha256'):
            p.run(source, self.base/'badout')

    def test_p112_normalized_and_raw_hashes(self):
        text = '天地玄黄 宇宙洪荒'
        raw = hashlib.sha256(text.encode()).hexdigest()
        norm = hashlib.sha256(text.replace(' ', '').lower().encode()).hexdigest()
        source = self.base/'p112.jsonl'
        data = {'text': text, 'group': 'g', 'sha256': norm, 'source_original_sha256': raw}
        source.write_text(json.dumps(data, ensure_ascii=False)+'\n', encoding='utf8')
        result = p.run(source, self.base/'good')
        self.assertEqual(result['input']['records'], 1)
        data['sha256'] = 'wrong'
        source.write_text(json.dumps(data, ensure_ascii=False)+'\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'normalized'):
            p.run(source, self.base/'badnorm')
        data['sha256'], data['source_original_sha256'] = norm, 'wrong'
        source.write_text(json.dumps(data, ensure_ascii=False)+'\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'sha256'):
            p.run(source, self.base/'badraw')


if __name__ == '__main__':
    unittest.main()
