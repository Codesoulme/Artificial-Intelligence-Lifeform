import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_p98_intro_review_queue as q


class QueueTests(unittest.TestCase):
    def test_transitive_components_and_history(self):
        base = ''.join(chr(0x4e00+i) for i in range(100))
        changed = base[:45] + '甲' + base[46:]
        changed_twice = changed[:65] + '乙' + changed[66:]
        data = [{'page_id':str(i), 'text':t} for i,t in enumerate([base, changed, changed_twice, '完全不同的文字内容' * 20])]
        sets, comp = q.components(data)
        self.assertEqual(comp[0], comp[2])
        self.assertNotEqual(comp[0], comp[3])
        historical = [{'id':'outside', 'text':base + '\n\nignored later paragraph'}]
        linked = q.related_indices(data, sets, historical)
        self.assertIn(0, linked)
        self.assertIn(comp[2], {comp[i] for i in linked})
        self.assertIn(3, q.related_indices(data, sets, [{'id':'3','text':'unrelated'}]))

    def test_selection_is_stable_and_distinct(self):
        candidates = [{'page_id':str(i), 'group':str(i), '_component':i//2} for i in range(12)]
        first = q.choose(candidates, 4, 'fixed')
        self.assertEqual(first, q.choose(list(reversed(candidates)), 4, 'fixed'))
        self.assertEqual(len({r['_component'] for r in first}), 4)
        with self.assertRaises(ValueError): q.choose(candidates, 7, 'fixed')

    def test_exact_hash_and_normalization(self):
        self.assertEqual(q.norm('Ａ，B！'), 'ab')
        self.assertNotEqual(q.text_hash('A'), q.text_hash('Ａ'))


if __name__ == '__main__': unittest.main()
