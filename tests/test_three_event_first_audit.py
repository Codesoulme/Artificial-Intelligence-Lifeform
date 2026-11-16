import unittest

from scripts.deepseek_spora_three_event_first_audit_20261117 import selected_order


class FirstEventPrototypeTests(unittest.TestCase):
    def test_first_marker_only_changes_previous_effective_selection(self):
        self.assertEqual(selected_order('小林最初喜欢什么饮料？', 'effective_previous'), 'effective_first')
        self.assertEqual(selected_order('小林更改前喜欢什么饮料？', 'effective_previous'), 'effective_previous')
        self.assertEqual(selected_order('小林最初喜欢什么饮料？', 'effective_latest'), 'effective_latest')

    def test_negated_earliest_marker_remains_an_unresolved_failure(self):
        # This shows why the prototype must remain isolated: the selected
        # first event contradicts the explicitly negated target.
        self.assertEqual(selected_order('不问最初，请说更改前喜欢什么饮料。', 'effective_previous'),
                         'effective_first')


if __name__ == '__main__':
    unittest.main()
