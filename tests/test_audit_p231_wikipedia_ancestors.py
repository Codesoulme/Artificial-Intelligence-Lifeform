import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import audit_p231_wikipedia_ancestors as audit


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode('utf8') + b'\n'


class AncestorAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=audit.ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.text = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        self.row = {'page_id': '42', 'char_start': 2, 'char_end': 38,
                    'text': self.text, 'text_sha256': audit.digest(self.text.encode())}

    def candidates(self, rows):
        raw = b''.join(encoded(row) for row in rows)
        (self.root / 'candidates.jsonl').write_bytes(raw)
        manifest = encoded({'files': {'candidates.jsonl': audit.digest(raw)},
                            'counts': {'retained_passages': len(rows)}})
        (self.root / 'manifest.json').write_bytes(manifest)
        return audit.digest(manifest), audit.digest(raw), len(rows)

    def test_identity_and_digests(self):
        args = self.candidates([self.row])
        self.assertEqual(audit.load_candidates(self.root, *args)[0][0], '42:2:38')
        with self.assertRaisesRegex(ValueError, 'manifest digest'):
            audit.load_candidates(self.root, '0' * 64, *args[1:])
        (self.root / 'candidates.jsonl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Candidate digest'):
            audit.load_candidates(self.root, *args)
        args = self.candidates([self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            audit.load_candidates(self.root, *args)
        args = self.candidates([{**self.row, 'text_sha256': '0' * 64}])
        with self.assertRaisesRegex(ValueError, 'identity or text digest'):
            audit.load_candidates(self.root, *args)

    def test_full_containment_counts_records_and_shared_prefix(self):
        prefix = 'abcdefghijklmnopqrstuvwxyz'
        rows = [('one', prefix + 'first'), ('two', prefix + 'second'), ('short', 'short')]
        sources = {'p28': self.root / 'p28', 'v5': self.root / 'v5'}
        sources['p28'].write_bytes(encoded({'text': prefix + 'first ' + prefix + 'first'}) +
                                   encoded({'text': prefix + 'SECOND'}) +
                                   encoded({'text': prefix + 'third'}))
        sources['v5'].write_bytes(encoded({'text': 'a SHORT passage'}))
        expected = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sources.items()}
        counts, metadata = audit.scan(rows, sources, expected)
        self.assertEqual(counts, {'p28': [1, 1, 0], 'v5': [0, 0, 1]})
        self.assertEqual(metadata['p28']['records'], 3)
        sources['v5'].write_bytes(encoded({'text': 'changed'}))
        with self.assertRaisesRegex(ValueError, 'source digest'):
            audit.scan(rows, sources, expected)

    def test_existing_output_refused_before_audit(self):
        output = self.root / 'output.json'
        output.write_text('original', encoding='utf8')
        with patch.object(audit, 'OUTPUT', output), patch.object(audit, 'audit') as run:
            with self.assertRaises(FileExistsError):
                audit.main()
            run.assert_not_called()
        self.assertEqual(output.read_text(encoding='utf8'), 'original')


if __name__ == '__main__':
    unittest.main()
