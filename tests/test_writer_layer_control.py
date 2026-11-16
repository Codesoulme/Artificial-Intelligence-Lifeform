import unittest

import torch
from scripts.deepseek_spora_writer_layer_control import create_cases, make_writer


class WriterLayerControlTests(unittest.TestCase):
    def test_all_name_value_split_combinations_are_present(self):
        rows = create_cases(('a',), ('b',), ('x', 'y'), ('z',))
        self.assertEqual(len(rows), 6)
        self.assertEqual(sum(r['name_split'] == 'train' and r['value_split'] == 'train' for r in rows), 2)
        self.assertEqual(sum(r['name_split'] == 'held' and r['value_split'] == 'held' for r in rows), 1)

    def test_rejects_overlapping_splits_and_duplicates(self):
        for args in ((('a',), ('a',), ('x',), ('z',)),
                     (('a',), ('b',), ('x',), ('x',)),
                     (('a', 'a'), ('b',), ('x',), ('z',))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                create_cases(*args)

    def test_identity_initialization_preserves_input(self):
        values = torch.randn(3, 4)
        torch.testing.assert_close(make_writer(4, 17, True)(values), values)
        self.assertFalse(torch.equal(make_writer(4, 17, False)(values), values))


if __name__ == '__main__':
    unittest.main()
