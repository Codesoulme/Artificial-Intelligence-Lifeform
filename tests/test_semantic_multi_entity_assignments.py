import unittest

from scripts.deepseek_spora_semantic_bridge_pilot import multi_entity_assignments, replace_assignment


class MultiEntityAssignmentTests(unittest.TestCase):
    def test_supervised_value_is_bound_to_each_target(self):
        for target in (0, 1, 2):
            for value in range(5):
                rows = multi_entity_assignments(target, value, 5)
                self.assertEqual(dict(rows)[target], value)
                self.assertEqual(len(dict(rows)), 3)
                self.assertEqual(len({v for _, v in rows}), 3)

    def test_counterfactual_changes_only_requested_memory(self):
        before = multi_entity_assignments(1, 3, 5)
        after = replace_assignment(before, 1, 4)
        self.assertEqual(dict(after)[1], 4)
        self.assertEqual(dict(after)[0], dict(before)[0])
        self.assertEqual(dict(after)[2], dict(before)[2])
        with self.assertRaises(ValueError):
            replace_assignment(before, 8, 4)

    def test_invalid_targets_and_bounds(self):
        with self.assertRaises(ValueError):
            multi_entity_assignments(3, 0, 5)
        with self.assertRaises(ValueError):
            multi_entity_assignments(1, 5, 5)


if __name__ == '__main__':
    unittest.main()
