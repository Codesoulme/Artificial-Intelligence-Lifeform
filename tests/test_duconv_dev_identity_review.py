import unittest
from unittest.mock import patch

from scripts import deepseek_spora_duconv_dev_identity_audit as audit


class DuConvDevIdentityReviewTests(unittest.TestCase):
    def test_reviewed_manifest_and_corrections(self):
        rows = [{'text': f'text-{i}', 'names': [f'name-{i}']} for i in range(48)]
        rows[24] = {'text': '唐·翰，他的作品。', 'names': []}
        manifest = audit.digest(''.join(audit.digest(row['text']) for row in rows))
        with patch.object(audit, 'MANIFEST_SHA', manifest):
            reviewed = audit.reviewed_sample(rows)
            self.assertEqual(len(reviewed), 46)
            self.assertEqual(reviewed[2]['reviewed_group'], 'multiple')
            self.assertIsNone(reviewed[2]['subject'])
            self.assertEqual(reviewed[24]['subject'], '唐·翰')
            self.assertEqual(sum(row['reviewed_group'] == 'single' for row in reviewed), 24)
            self.assertEqual(sum(row['reviewed_group'] == 'none' for row in reviewed), 9)
            self.assertEqual(sum(row['reviewed_group'] == 'multiple' for row in reviewed), 13)
            rows[0]['text'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'sample changed'):
                audit.reviewed_sample(rows)


if __name__ == '__main__':
    unittest.main()
