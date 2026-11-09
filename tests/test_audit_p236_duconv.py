import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_p236_duconv import audit, conversation_key


class DuConvAuditTests(unittest.TestCase):
    def test_dedup_and_metadata_only_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.zip'
            rows = [{'conversation': ['你好', '您好'], 'goal': [], 'knowledge': []},
                    {'conversation': ['你好', '您好'], 'goal': [], 'knowledge': []},
                    {'conversation': ['我的邮箱 a@example.com', '知道了'], 'goal': [], 'knowledge': []},
                    {'conversation': []}]
            with ZipFile(path, 'w') as z:
                z.writestr('DuConv/train.txt', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
                z.writestr('DuConv/test_1.txt', 'not json; must not be read')
            result, sample = audit(path, sample_size=2)
            self.assertEqual(result['counts']['rows'], 4)
            self.assertEqual(result['counts']['exact_duplicate_conversations'], 1)
            self.assertEqual(result['counts']['invalid_conversation'], 1)
            self.assertEqual(result['heuristic_flagged_conversations']['contact'], 1)
            self.assertEqual(len(sample), 2)
            self.assertEqual(sample, audit(path, sample_size=2)[1])

    def test_normalized_key_and_missing_member(self):
        self.assertEqual(conversation_key(['Ａ']), conversation_key(['a']))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.zip'
            with ZipFile(path, 'w') as z:
                z.writestr('DuConv/test_1.txt', 'data')
            with self.assertRaises(ValueError):
                audit(path, sample_size=1)


if __name__ == '__main__':
    unittest.main()
