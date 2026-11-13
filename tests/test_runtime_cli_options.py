import unittest

from scripts import deepseek_spora_runtime_chat as runtime


class RuntimeCliOptionTests(unittest.TestCase):
    def test_default_floor_is_unchanged(self):
        self.assertEqual(runtime.READ_INTENT_RECOVERY_FLOOR, 2.0)

    def test_default_intent_uses_v4_after_v7_predicate_regression(self):
        self.assertEqual(runtime.INTENT.name, 'deepseek-spora-read-intent-gate-v4-20261111')


if __name__ == '__main__':
    unittest.main()
