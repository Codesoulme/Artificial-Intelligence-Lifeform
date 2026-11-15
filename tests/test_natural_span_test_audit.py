import unittest
from unittest.mock import patch

from scripts import deepseek_spora_natural_span_test_audit as audit


class NaturalSpanTestAuditTests(unittest.TestCase):
    def test_reviewed_labels_and_manifest(self):
        rows = [{'text': f'text-{index}', 'names': [f'name-{index}']} for index in range(36)]
        rows[15] = {'text': '我知道一个明星，名字是朱迪·福斯特。', 'names': []}
        manifest = audit.dev.digest(''.join(audit.dev.digest(row['text']) for row in rows))
        with patch.object(audit, 'MANIFEST_SHA', manifest):
            reviewed = audit.reviewed_sample(rows)
            self.assertEqual(len(reviewed), 36)
            self.assertEqual(sum(row['reviewed_group'] == 'single' for row in reviewed), 11)
            self.assertEqual(sum(row['reviewed_group'] == 'none' for row in reviewed), 11)
            self.assertEqual(sum(row['reviewed_group'] == 'multiple' for row in reviewed), 14)
            self.assertEqual(reviewed[15]['subject'], '朱迪·福斯特')
            rows[0]['text'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'sample changed'):
                audit.reviewed_sample(rows)


if __name__ == '__main__':
    unittest.main()
