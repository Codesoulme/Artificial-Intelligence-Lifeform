import json
from pathlib import Path
import tempfile
import unittest

from scripts import sample_p230_wikipedia_body as sample


class BodySampleTests(unittest.TestCase):
    def test_one_per_page_reproducible_and_tamper_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root/'source'
            candidate.mkdir()
            rows = []
            for page, start in [('1', 0), ('1', 5), ('2', 0)]:
                text = '中文内容'
                rows.append({'page_id': page, 'char_start': start, 'char_end': start+len(text),
                             'text': text, 'text_sha256': sample.sha(text.encode())})
            raw = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode()
            (candidate/'candidates.jsonl').write_bytes(raw)
            (candidate/'manifest.json').write_text(json.dumps({'status': 'pending_review_not_approved_for_training',
                'files': {'candidates.jsonl': sample.sha(raw)}, 'counts': {'passages': 3}}), encoding='utf8')
            one = sample.build(candidate, root/'one', count=2)
            two = sample.build(candidate, root/'two', count=2)
            self.assertEqual(one, two)
            self.assertEqual(len({x['page_id'] for x in json.loads((root/'one'/'sample.json').read_text(encoding='utf8'))}), 2)
            with self.assertRaises(FileExistsError):
                sample.build(candidate, root/'one', count=2)
            (candidate/'candidates.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'digest'):
                sample.build(candidate, root/'bad', count=2)
            self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
