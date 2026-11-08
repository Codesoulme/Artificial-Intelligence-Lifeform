import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_p101_intro_review_queue as q
import build_p99_intro_review_queue as p99


class ThirdQueueTests(unittest.TestCase):
    def test_whole_components_are_removed_and_seed_is_new(self):
        candidates = [dict(page_id=str(i), group=str(i), _component=i // 2) for i in range(16)]
        remaining = q.exclude_components(candidates, {1, 3})
        selected = q.choose(remaining, 6, q.SEED)
        self.assertEqual({r['_component'] for r in selected}, {0, 2, 4, 5, 6, 7})
        self.assertEqual(selected, q.choose(list(reversed(remaining)), 6, q.SEED))
        self.assertNotIn(q.SEED, (q.p98.SEED, p99.SEED))

    def test_path_rejects_existing_and_traversal(self):
        with tempfile.TemporaryDirectory() as d, patch.object(q, 'ROOT', Path(d)):
            (Path(d) / 'runs' / 'used').mkdir(parents=True)
            self.assertEqual(q.output_path('fresh'), Path(d) / 'runs' / 'fresh')
            for name in ('used', '', '.', '..', '../other', 'a/b', r'a\b', 'C:other'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    q.output_path(name)

    def test_p99_manifest_is_pinned(self):
        with tempfile.TemporaryDirectory() as d, patch.object(q, 'ROOT', Path(d)):
            directory = Path(d) / 'runs/wikimedia-intro-review-p99-v1-repro'
            directory.mkdir(parents=True)
            (directory / 'manifest.json').write_text('{}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                q.load_p99([], [], [], {})

    def test_p99_shards_and_records_verified(self):
        with tempfile.TemporaryDirectory() as d, patch.object(q, 'ROOT', Path(d)):
            base = Path(d) / 'runs/wikimedia-intro-review-p99-v1-repro'
            base.mkdir(parents=True)
            rows = [dict(page_id=str(i), group=f'g{i}', url=f'u{i}', title=f't{i}', text=f'text{i}') for i in range(400)]
            comp = list(range(400)); labels = {i: f'g{i}' for i in comp}
            files = {}
            for n in range(1, 5):
                items = [dict(page_id=r['page_id'], group=r['group'], source_url=r['url'], title=r['title'], text=r['text'], text_sha256=q.text_hash(r['text']), similarity_component=r['group'], review_status='pending', review=None) for r in rows[(n-1)*100:n*100]]
                shard = dict(status='pending_not_approved_for_training', source_sha256=q.RAW, train_sha256=q.V3, screened_sha256=q.SCREEN, samples=items)
                path = base / f'review-{n:02}.json'
                path.write_text(json.dumps(shard), encoding='utf8'); files[path.name] = q.digest(path)
            manifest = dict(status='pending_not_approved_for_training', seed=p99.SEED, raw_source_sha256=q.RAW, train_v1_sha256=q.V1, train_v3_sha256=q.V3, p77_screened_sha256=q.SCREEN, p98_manifest_sha256=q.P98_MANIFEST_SHA256, counts=dict(selected=400, shards=4, per_shard=100), files=files)
            mp = base / 'manifest.json'; mp.write_text(json.dumps(manifest), encoding='utf8')
            with patch.object(q, 'P99_MANIFEST_SHA256', q.digest(mp)):
                self.assertEqual(len(q.load_p99(rows, rows, comp, labels)), 400)
                path = base / 'review-01.json'
                data = json.loads(path.read_text(encoding='utf8'))
                data['samples'][0]['text'] = 'changed'
                path.write_text(json.dumps(data), encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    q.load_p99(rows, rows, comp, labels)
                manifest['files'][path.name] = q.digest(path)
                mp.write_text(json.dumps(manifest), encoding='utf8')
                with patch.object(q, 'P99_MANIFEST_SHA256', q.digest(mp)):
                    with self.assertRaisesRegex(ValueError, 'record mismatch'):
                        q.load_p99(rows, rows, comp, labels)


if __name__ == '__main__':
    unittest.main()
