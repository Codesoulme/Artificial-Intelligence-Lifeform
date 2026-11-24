import unittest

from spora.research_version_menu import ORDERS, menu_evidence, relative_menu_allowed


class VersionMenuTests(unittest.TestCase):
    def test_semantic_choice_survives_relabeling(self):
        scores = [[8 if option == 1 else -2 for option in order] for order in ORDERS]
        result = menu_evidence(scores)
        self.assertEqual(result['option'], 1)
        self.assertEqual(result['votes'], [1, 1, 1, 1])
        self.assertEqual(result['agreement'], 4)
        self.assertEqual(set(result['positions']), {0, 1, 2, 3})

    def test_always_choosing_the_first_letter_is_not_semantic_agreement(self):
        result = menu_evidence([[8, 0, 0, 0]] * 4)
        self.assertEqual(result['positions'], [0, 0, 0, 0])
        self.assertEqual(result['votes'], [0, 1, 2, 3])
        self.assertEqual(result['agreement'], 1)
        self.assertEqual(result['margin'], 0)

    def test_literal_coordinates_and_scope_blocks_bypass_menu(self):
        self.assertTrue(relative_menu_allowed('照手头的那张来，霍盈那项是什么？', 0, 'unresolved_time'))
        self.assertFalse(relative_menu_allowed('２０２６－１０－１２那张是什么？', 0, 'timeline'))
        self.assertFalse(relative_menu_allowed('第二版的饮品是什么？', 0, 'timeline'))
        self.assertFalse(relative_menu_allowed('不查记录。', 0, 'outside_lookup_scope'))
        self.assertFalse(relative_menu_allowed('手头有没有记录？', None, 'unresolved_address'))
