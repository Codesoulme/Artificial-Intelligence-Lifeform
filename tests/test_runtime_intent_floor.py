import unittest
from types import SimpleNamespace

from scripts import deepseek_spora_runtime_chat as runtime


class RuntimeIntentFloorTests(unittest.TestCase):
    def test_chat_uses_default_floor_without_global_mutation(self):
        chat = object.__new__(runtime.SporaChat)
        chat.read_intent_recovery_floor = runtime.READ_INTENT_RECOVERY_FLOOR
        self.assertEqual(chat.read_intent_recovery_floor, 2.0)
        self.assertEqual(runtime.READ_INTENT_RECOVERY_FLOOR, 2.0)

    def test_legacy_test_double_falls_back_to_module_floor(self):
        chat = object.__new__(runtime.SporaChat)
        self.assertEqual(getattr(chat, 'read_intent_recovery_floor', runtime.READ_INTENT_RECOVERY_FLOOR), 2.0)


if __name__ == '__main__':
    unittest.main()
