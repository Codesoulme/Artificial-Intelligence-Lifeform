import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_belle_prefix_overlap import audit


class BelleOverlapTests(unittest.TestCase):
    def test_exact_overlap_and_partial_last_line(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / 'train.jsonl'
            old.write_text(json.dumps({'messages': [{'role': 'user', 'content': '你好 世 界'},
                                                     {'role': 'assistant', 'content': '好。'}]}, ensure_ascii=False) + '\n', encoding='utf-8')
            prefix = root / 'prefix.bin'
            row = {'instruction': 'Human:你好世界\nAssistant:过去\nHuman:继续\nAssistant:', 'output': '好。'}
            prefix.write_bytes((json.dumps(row, ensure_ascii=False) + '\n{"instruction":').encode('utf-8'))
            result = audit(prefix, old)
            self.assertEqual(result['complete_rows'], 1)
            self.assertEqual(result['rows_first_prompt_matching_train'], 1)
            self.assertEqual(result['rows_output_matching_train'], 1)
            prefix.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
            self.assertEqual(audit(prefix, old)['complete_rows'], 1)


if __name__ == '__main__':
    unittest.main()
