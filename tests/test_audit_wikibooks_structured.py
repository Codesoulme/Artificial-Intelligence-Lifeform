import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_wikibooks_structured as audit


class AuditTests(unittest.TestCase):
    def test_binds_sample_and_checks_source_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample, output = root/'sample.json', root/'audit.json'
            text = '完整段落。\n\n正文<ref>来源</ref>。'
            rows = [{'page_id': '1', 'raw_wikitext': text,
                     'raw_sha256': hashlib.sha256(text.encode()).hexdigest()}]
            raw = json.dumps(rows, ensure_ascii=False).encode()
            sample.write_bytes(raw)
            with patch.object(audit, 'SAMPLE_SHA', hashlib.sha256(raw).hexdigest()):
                result = audit.audit(sample, output)
                self.assertEqual(result['counts']['review_candidate_paragraphs'], 1)
                self.assertEqual(result['counts']['quarantined_paragraphs'], 1)
                self.assertEqual(result['risk_occurrences']['citation'], 1)
                with self.assertRaises(FileExistsError):
                    audit.audit(sample, output)
                sample.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'digest'):
                    audit.audit(sample, root/'changed.json')
                self.assertFalse((root/'changed.json').exists())


if __name__ == '__main__':
    unittest.main()
