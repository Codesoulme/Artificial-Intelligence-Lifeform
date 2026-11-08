import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_p99_intro_review_queue as q


class FollowupQueueTests(unittest.TestCase):
    def test_excludes_entire_component_before_selection(self):
        candidates = [dict(page_id=str(i), group=str(i), _component=i // 2) for i in range(12)]
        eligible = q.exclude_components(candidates, {1, 3})
        selected = q.choose(eligible, 4, q.SEED)
        self.assertEqual({r['_component'] for r in selected}, {0, 2, 4, 5})
        self.assertEqual(selected, q.choose(list(reversed(eligible)), 4, q.SEED))
        self.assertNotEqual(q.SEED, q.p98.SEED)

    def test_output_path_rejects_traversal_and_existing_directory(self):
        with tempfile.TemporaryDirectory() as d, patch.object(q, 'ROOT', Path(d)):
            (Path(d) / 'runs' / 'used').mkdir(parents=True)
            self.assertEqual(q.output_path('fresh'), Path(d) / 'runs' / 'fresh')
            for name in ('used', '..', '../other', 'nested/fresh', r'nested\fresh', 'C:other'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    q.output_path(name)

    def test_p98_manifest_is_pinned(self):
        with tempfile.TemporaryDirectory() as d, patch.object(q, 'ROOT', Path(d)):
            directory = Path(d) / 'runs/wikimedia-intro-review-p98-v1'
            directory.mkdir(parents=True)
            (directory / 'manifest.json').write_text('{}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                q.load_p98([], [], [], {})


if __name__ == '__main__':
    unittest.main()
