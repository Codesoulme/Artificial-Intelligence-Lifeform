import unittest
from scripts.deepseek_spora_dialogue_values_memory_pilot_20261117 import selected_values


class DialogueValueSelectionTests(unittest.TestCase):
    def test_selects_exact_after_values_and_rejects_missing_duplicate(self):
        rows = [{'session': s, 'condition': 'after', 'answer': value}
                for s, value in ((151, '踢球'), (93, '果汁'), (164, '教师'))]
        self.assertEqual(selected_values({'rows': rows}), ('踢球', '果汁', '教师'))
        with self.assertRaises(ValueError):
            selected_values({'rows': rows[:-1]})
        with self.assertRaises(ValueError):
            selected_values({'rows': rows + [rows[0]]})
