import unittest

from scripts.audit_crosswoz_train import git_blob_sha, summarize


class CrossWOZAuditTests(unittest.TestCase):
    def test_statistics_count_messages_and_risk_indicators(self):
        data = {'x': {'type': 'test', 'messages': [
            {'role': 'usr', 'content': '你好'},
            {'role': 'sys', 'content': '网址 https://example.org'}]}}
        result = summarize(data)
        self.assertEqual(result['dialogues'], 1)
        self.assertEqual(result['messages'], 2)
        self.assertEqual(result['risk_pattern_counts']['url_pattern_messages'], 1)
        data['x']['messages'][0]['role'] = 'unknown'
        with self.assertRaises(ValueError):
            summarize(data)

    def test_git_blob_hash(self):
        self.assertEqual(git_blob_sha(b''), 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391')


if __name__ == '__main__':
    unittest.main()
