import unittest
from scripts.deepseek_kernel_multirelation_pair_20261117 import build_cases, RELATIONS


class MultiRelationPairTests(unittest.TestCase):
    def test_every_training_case_has_counterfactual_same_question(self):
        train, _ = build_cases()
        for item in train:
            other = [r for r in train if r['name'] == item['name'] and r['relation'] == item['relation']
                     and r['value'] != item['value']]
            self.assertEqual(len(other), 1)
            self.assertEqual(item['question'], other[0]['question'])
            self.assertNotEqual(item['answer'], other[0]['answer'])
            self.assertEqual(len(RELATIONS[item['relation']]['train']), 2)
