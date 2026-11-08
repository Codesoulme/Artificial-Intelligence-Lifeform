import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audit', ROOT / 'scripts/audit_story_prefix_concentration.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class BoundaryAuditTests(unittest.TestCase):
    def test_lengths_whitespace_and_exact_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'input.jsonl'
            rows = [{'text': '甲' * 20 + '乙'}, {'text': '甲 ' * 20 + '丙'}, {'text': '短'}]
            source.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows), encoding='utf-8')
            result = audit.audit(source)
            self.assertEqual(result['documents'], 3)
            self.assertEqual(result['exact_text']['unique_strings'], 3)
            self.assertEqual(result['prefix']['20']['eligible_documents'], 2)
            self.assertEqual(result['prefix']['20']['documents_in_shared_groups'], 2)
            self.assertEqual(result['prefix']['20']['duplicate_occurrences_beyond_first'], 1)
            self.assertEqual(result['prefix']['40']['eligible_documents'], 0)


if __name__ == '__main__':
    unittest.main()
