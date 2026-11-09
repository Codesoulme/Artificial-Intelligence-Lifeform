import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import evaluate_reviewed_wiki_dev as evaluation


class ReviewedWikiDevelopmentTests(unittest.TestCase):
    def test_reads_only_bound_development_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'wiki'
            directory.mkdir()
            text = '一段开发文本。'
            row = {'text': text, 'text_sha256': hashlib.sha256(text.encode('utf8')).hexdigest()}
            raw = (json.dumps(row, ensure_ascii=False)+'\n').encode('utf8')
            (directory/'development.jsonl').write_bytes(raw)
            manifest = json.dumps({'files': {'development.jsonl': hashlib.sha256(raw).hexdigest()}}).encode()
            (directory/'manifest.json').write_bytes(manifest)
            sources = (('wiki99', 'wiki', hashlib.sha256(manifest).hexdigest()),)
            with patch.object(evaluation, 'SOURCES', sources):
                rows, hashes = evaluation.read_development(root)
                self.assertEqual(rows[0]['text'], text)
                self.assertEqual(hashes['wiki99']['data_sha256'], hashlib.sha256(raw).hexdigest())
                (directory/'development.jsonl').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'data changed'):
                    evaluation.read_development(root)


if __name__ == '__main__':
    unittest.main()
