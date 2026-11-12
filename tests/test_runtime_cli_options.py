import unittest

from scripts import deepseek_spora_runtime_chat as runtime


class RuntimeCliOptionTests(unittest.TestCase):
    def test_default_floor_is_unchanged(self):
        self.assertEqual(runtime.READ_INTENT_RECOVERY_FLOOR, 2.0)


if __name__ == '__main__':
    unittest.main()
