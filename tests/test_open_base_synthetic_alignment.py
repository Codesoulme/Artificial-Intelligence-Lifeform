import unittest

import torch

from scripts.open_base_synthetic_alignment import choice_ids, make_state


class SyntheticAlignmentTests(unittest.TestCase):
    def test_state_reproducibility_swap_and_equal_norm(self):
        a = make_state(8, 5, 'cpu')
        b = make_state(8, 5, 'cpu', swap=True)
        c = make_state(8, 6, 'cpu')
        empty = make_state(8, 5, 'cpu', empty=True)
        for x, y, z, zero in zip(a.current, b.current, c.current, empty.current):
            self.assertTrue(torch.equal(x.flip(0), y))
            self.assertFalse(torch.equal(x, z))
            self.assertTrue(torch.allclose(x.flatten(1).norm(dim=1), torch.ones(2)))
            self.assertEqual(int(torch.count_nonzero(zero)), 0)

    def test_choice_ids_require_two_distinct_single_tokens(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {'1': [1], '2': [2]}[text]
        self.assertEqual(choice_ids(Tokenizer()), [1, 2])
        class Invalid:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]
        with self.assertRaises(ValueError):
            choice_ids(Invalid())


if __name__ == '__main__':
    unittest.main()
