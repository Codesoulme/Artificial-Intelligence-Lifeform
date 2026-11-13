import unittest

from scripts.deepseek_spora_original_layer_name_router import locator_only_rows


class OriginalNameBoundaryTrainingTests(unittest.TestCase):
    def test_long_name_is_labeled_without_binding_a_key(self):
        rows = locator_only_rows([('阿远鹏', '请查阿远鹏的饮品。')])
        self.assertEqual(rows[0]['name'], '阿远鹏')
        self.assertIsNone(rows[0]['key'])
        self.assertEqual(rows[0]['group'], 'train_locator_only')

    def test_registered_or_ambiguous_names_are_rejected(self):
        for item in [('阿远', '阿远喜欢喝什么？'), ('阿远鹏', '阿远喜欢喝什么？'),
                     ('阿远鹏', '阿远鹏和阿远鹏喜欢什么？')]:
            with self.assertRaises(ValueError):
                locator_only_rows([item])


if __name__ == '__main__':
    unittest.main()
