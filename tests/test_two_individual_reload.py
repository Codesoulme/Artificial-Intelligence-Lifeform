import unittest

from scripts.deepseek_two_individual_reload import OUTPUT


class TwoIndividualReloadTests(unittest.TestCase):
    def test_output_is_isolated_under_runs(self):
        self.assertTrue(OUTPUT.is_absolute())
        self.assertEqual(OUTPUT.parent.name, 'runs')
        self.assertEqual(OUTPUT.name, 'deepseek-two-individual-reload-20261110')


if __name__ == '__main__':
    unittest.main()
