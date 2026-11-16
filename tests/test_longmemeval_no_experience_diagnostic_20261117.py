import json
import tempfile
import unittest
from pathlib import Path

from scripts.longmemeval_no_experience_diagnostic_20261117 import CASE_IDS, select_cases


class NoExperienceDiagnosticTests(unittest.TestCase):
    def test_selects_fixed_cases_without_reading_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.jsonl'
            rows = [{'case_id': case_id, 'question': 'question', 'sessions': []}
                    for case_id in reversed(CASE_IDS)]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            self.assertEqual([row['case_id'] for row in select_cases(path)], list(CASE_IDS))
            self.assertFalse((Path(folder) / 'gold.private.jsonl').exists())

    def test_rejects_missing_and_duplicate_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'input.jsonl'
            path.write_text(json.dumps({'case_id': CASE_IDS[0]}) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                select_cases(path)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'case_id': CASE_IDS[0]}) + '\n')
            with self.assertRaises(ValueError):
                select_cases(path)
