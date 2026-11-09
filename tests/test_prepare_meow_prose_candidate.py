import json
import unittest

from scripts.prepare_meow_prose_candidate import parse_paragraphs


class ProseCandidateTests(unittest.TestCase):
    def test_order_and_complete_source_are_required(self):
        rows = [{'id': f'meow-v2-p{i:02d}', 'para_id': i, 'text': f'段落{i}'} for i in range(1, 34)]
        raw = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows).encode('utf8')
        self.assertEqual(parse_paragraphs(raw).splitlines(), [row['text'] for row in rows])
        with self.assertRaises(ValueError):
            parse_paragraphs(b'')
        rows[1]['para_id'] = 3
        bad = '\n'.join(json.dumps(row, ensure_ascii=False) for row in rows).encode('utf8')
        with self.assertRaises(ValueError):
            parse_paragraphs(bad)


if __name__ == '__main__':
    unittest.main()
