import unittest

from spora.paired_group_sample import select


class PairedGroupSampleTests(unittest.TestCase):
    def test_stable_cap_and_pair_alignment(self):
        a = [{'pair_id': str(i), 'group': 'big' if i < 6 else 'small',
              'tokens': [1, 5, i + 10], 'mask': [False, False, True]} for i in range(8)]
        b = [{'pair_id': r['pair_id'], 'group': r['group'],
              'tokens': [1, r['tokens'][-1]], 'mask': [False, True]} for r in reversed(a)]
        x, y, meta = select(a, b, seed=3, max_per_group=2, length=8)
        self.assertEqual(len(x), 4)
        self.assertEqual([r['pair_id'] for r in x], [r['pair_id'] for r in y])
        self.assertEqual(meta['largest_group_selected'], 2)
        self.assertEqual(select(list(reversed(a)), b, seed=3, max_per_group=2, length=8), (x, y, meta))

    def test_rejects_mismatch_even_when_group_would_be_capped(self):
        a = [{'pair_id': str(i), 'group': 'g', 'tokens': [1, i + 2], 'mask': [False, True]} for i in range(3)]
        b = [dict(r) for r in a]
        b[0] = dict(b[0], tokens=[1, 99])
        with self.assertRaises(ValueError):
            select(a, b, seed=1, max_per_group=1, length=5)


if __name__ == '__main__':
    unittest.main()
