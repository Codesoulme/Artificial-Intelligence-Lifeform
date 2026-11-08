import copy
import unittest

from scripts.build_p108_reviewed_pairs import select


class ReviewedPairsTests(unittest.TestCase):
    def test_binding_and_review_coverage(self):
        a = [{'pair_id': str(i), 'group': str(i), 'tokens': [1, 2, i + 3],
              'mask': [False, False, True]} for i in range(3)]
        b = [{'pair_id': row['pair_id'], 'group': row['group'], 'tokens': [1, row['tokens'][-1]],
              'mask': [False, True]} for row in a]
        meta = [{'pair_id': row['pair_id'], 'group': row['group']} for row in a]
        queue = copy.deepcopy(meta)
        review = [dict(row, 判定='通过' if i != 1 else '存疑') for i, row in enumerate(meta)]
        x, y, z = select(queue, review, a, b, meta)
        self.assertEqual([r['pair_id'] for r in x], ['0', '2'])
        self.assertEqual([r['pair_id'] for r in y], [r['pair_id'] for r in z])
        with self.assertRaises(ValueError):
            select(queue, review[:-1], a, b, meta)
        broken = copy.deepcopy(b)
        broken[1]['tokens'][-1] = 99
        with self.assertRaises(ValueError):
            select(queue, review, a, broken, meta)
        broken = copy.deepcopy(review)
        broken[0]['group'] = 'wrong'
        with self.assertRaises(ValueError):
            select(queue, broken, a, b, meta)


if __name__ == '__main__':
    unittest.main()
