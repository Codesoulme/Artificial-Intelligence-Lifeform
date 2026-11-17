import unittest
from scripts.deepseek_kernel_multirelation_train_20261117 import build_cases, NAMES, HELD_NAME, RELATIONS


class KernelMultiRelationTests(unittest.TestCase):
    def test_identity_and_value_splits(self):
        train, held = build_cases()
        self.assertEqual((len(train), len(held)), (12, 15))
        self.assertTrue(all(c['name'] in NAMES and c['value'] in RELATIONS[c['relation']]['train'] for c in train))
        self.assertTrue(all(c['name'] == HELD_NAME or c['value'] == RELATIONS[c['relation']]['held'] for c in held))
        self.assertTrue(all(c['value'] not in c['question'] and c['value'] in c['answer'] for c in train + held))
