import unittest

from scripts.deepseek_mmc_choice_baseline import balanced_subset, make_prompt


class MMCChoiceTests(unittest.TestCase):
    def test_balances_without_reordering_or_duplicating(self):
        rows = [{'label': 1, 'n': 1}, {'label': 1, 'n': 2},
                {'label': 0, 'n': 3}, {'label': 0, 'n': 4}]
        self.assertEqual([r['n'] for r in balanced_subset(rows, 1)], [1, 3])
        with self.assertRaises(ValueError):
            balanced_subset(rows, 3)

    def test_order_changes_choice_meaning(self):
        self.assertIn('1.同一人 2.不同人', make_prompt('text', False))
        self.assertIn('1.不同人 2.同一人', make_prompt('text', True))
        with self.assertRaises(ValueError):
            make_prompt('text', 1)


if __name__ == '__main__':
    unittest.main()
