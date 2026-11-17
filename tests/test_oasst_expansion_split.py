import unittest
from scripts.prepare_oasst_context_expansion_20261118 import split_candidates


class SplitTests(unittest.TestCase):
    def test_tree_and_current_text_do_not_cross_partitions(self):
        rows = [{'group': str(i), 'messages': [{'content': f'question{i//2}'}]} for i in range(20)]
        splits = split_candidates(rows, {'0'}, (2, 2, 2))
        groups = [r['group'] for part in splits.values() for r in part]
        texts = [r['messages'][-1]['content'] for part in splits.values() for r in part]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(len(texts), len(set(texts)))
        self.assertNotIn('0', groups)
        self.assertEqual(splits, split_candidates(list(reversed(rows)), {'0'}, (2, 2, 2)))

    def test_not_enough_data_fails(self):
        with self.assertRaises(ValueError):
            split_candidates([], set())
