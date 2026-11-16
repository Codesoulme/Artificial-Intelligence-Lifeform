import unittest

from scripts.deepseek_spora_shared_joint_counterfactual import make_assignments


class CounterfactualDesignTests(unittest.TestCase):
    def test_one_fact_changes_and_other_facts_remain_identical(self):
        before, after = make_assignments()
        self.assertEqual(len(before), 4)
        self.assertEqual([name for name, _ in before], [name for name, _ in after])
        self.assertNotEqual(before[0][1], after[0][1])
        self.assertEqual(before[1:], after[1:])


if __name__ == '__main__':
    unittest.main()
