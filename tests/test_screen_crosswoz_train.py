import unittest

from scripts.screen_crosswoz_train import flag_dialogue, screen


class CrossWOZScreenTests(unittest.TestCase):
    def test_role_sensitive_flags_and_counts(self):
        dialogue = {'type': 'fixture', 'messages': [
            {'role': 'usr', 'content': '帮我查一下价格'},
            {'role': 'sys', 'content': '我查了一下，地址不明，电话 010-12345678'}]}
        flags = flag_dialogue(dialogue)
        self.assertIn('query_claim', flags)
        self.assertIn('phone', flags)
        self.assertIn('address_terms', flags)
        rows, counts = screen({'id': dialogue})
        self.assertEqual(rows[0]['status'], 'unreviewed_not_approved')
        self.assertEqual(counts['phone'], 1)
        self.assertNotIn('content', rows[0])

    def test_user_query_request_is_not_system_claim(self):
        dialog = {'messages': [{'role': 'usr', 'content': '我查了价格'},
                               {'role': 'sys', 'content': '不知道'}]}
        self.assertNotIn('query_claim', flag_dialogue(dialog))


if __name__ == '__main__':
    unittest.main()
