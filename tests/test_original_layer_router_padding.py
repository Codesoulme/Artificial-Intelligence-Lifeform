import unittest

import torch

from scripts.deepseek_spora_original_layer_name_router import unpad_selection


class OriginalLayerRouterPaddingTests(unittest.TestCase):
    def test_right_padding_is_removed_without_moving_selection(self):
        selection = torch.tensor([False, True, True, False, False])
        self.assertTrue(torch.equal(unpad_selection(selection, 3), torch.tensor([False, True, True])))

    def test_selected_padding_is_rejected(self):
        with self.assertRaises(ValueError):
            unpad_selection(torch.tensor([False, True, False, True]), 3)

    def test_invalid_length_and_dtype_are_rejected(self):
        for length in (0, 5, True):
            with self.assertRaises(ValueError):
                unpad_selection(torch.tensor([False, True, False, False]), length)
        with self.assertRaises(ValueError):
            unpad_selection(torch.tensor([0, 1, 0]), 3)


if __name__ == '__main__':
    unittest.main()
