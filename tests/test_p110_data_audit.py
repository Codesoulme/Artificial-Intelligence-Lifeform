import hashlib
import io
import json
import unittest
from unittest.mock import Mock

from scripts.p110_data_audit import audit, signals


class AuditTests(unittest.TestCase):
    def test_signals_overlap_and_are_only_cues(self):
        flags = signals('问：如何处理？\n答：首先检查。其次记录。最后复核。')
        self.assertTrue(flags['question_mark'])
        self.assertTrue(flags['role_marker'])
        self.assertTrue(flags['template_cues'])
        self.assertFalse(flags['prose_candidate'])

    def test_deterministic_aggregate_and_duplicates(self):
        rows = []
        for i in range(30):
            text = '示例文字。' * (i % 4 + 1)
            normalized = text.lower()
            rows.append(dict(text=text, sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                             group=str(i % 4), source_byte_bucket=i % 10))
        payload = b''.join((json.dumps(x, ensure_ascii=False) + '\n').encode() for x in rows)
        path = Mock()
        path.open.side_effect = lambda *args: io.BytesIO(payload)
        a = audit(path, 10, 123)
        b = audit(path, 10, 123)
        self.assertEqual(a, b)
        self.assertEqual(a['population'], 30)
        self.assertEqual(a['duplicate_excess'], 26)
        self.assertEqual(a['signals']['duplicate_digest_in_population'], 10)
        self.assertEqual(a['signals']['digest_mismatch'], 0)
        self.assertNotIn('示例文字', json.dumps(a, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
