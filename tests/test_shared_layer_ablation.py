import unittest
from scripts.deepseek_spora_shared_layer_ablation import intervention


class LayerAblationDesignTests(unittest.TestCase):
    def test_each_intervention_changes_only_its_own_fact(self):
        for index in range(4):
            old, new = intervention(index)
            self.assertEqual([i for i in range(4) if old[i] != new[i]], [index])
            self.assertEqual(old[index][0], new[index][0])
        with self.assertRaises(ValueError):
            intervention(4)


if __name__ == '__main__':
    unittest.main()
