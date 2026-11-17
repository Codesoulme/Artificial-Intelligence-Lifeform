import unittest
from scripts.deepseek_kernel_natural_values_probe_20261117 import build_facts


class KernelNaturalValuesTests(unittest.TestCase):
    def test_values_remain_relation_specific(self):
        facts = build_facts(('跑步', '黄酒', '教师'))
        self.assertEqual(len(facts), 3)
        for index, value in enumerate(('跑步', '黄酒', '教师')):
            self.assertIn(value, facts[index])
            self.assertTrue(all(value not in f for j, f in enumerate(facts) if j != index))

    def test_rejects_invalid_values(self):
        for values in (('茶',), ('茶', '', '教师')):
            with self.assertRaises(ValueError):
                build_facts(values)
