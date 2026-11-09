import bz2
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_p226_validation as prep


class ValidationTests(unittest.TestCase):
    def test_excludes_previous_work_and_binds_raw_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, candidate, old = root/'dump.bz2', root/'candidate.jsonl', root/'old.json'
            text = '这是一段完整正文。'
            pages = ''.join(f'<page><title>教材{i}</title><id>{i}</id><revision><id>{i+10}</id>'
                            f'<text>{text}</text></revision></page>' for i in (1, 2))
            source.write_bytes(bz2.compress(f'<mediawiki xmlns="urn:test">{pages}</mediawiki>'.encode()))
            rows = [{'page_id': str(i), 'revision_id': str(i+10), 'title': f'教材{i}',
                     'work_sha256': f'work{i}', 'raw_sha256': prep.sha(text.encode())} for i in (1, 2)]
            candidate.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
            old.write_text(json.dumps([{'work_sha256': 'work1'}]), encoding='utf8')
            with patch.object(prep, 'EXPECTED_SHA1', hashlib.sha1(source.read_bytes()).hexdigest()), \
                 patch.object(prep, 'CANDIDATE_SHA', prep.sha(candidate.read_bytes())), \
                 patch.object(prep, 'OLD_SAMPLE_SHA', prep.sha(old.read_bytes())):
                result = prep.build(source, candidate, old, root/'out', count=1)
                self.assertEqual(result['pages'], 1)
                sample = json.loads((root/'out'/'review-sample.json').read_text(encoding='utf8'))
                self.assertEqual(sample[0]['page_id'], '2')
                self.assertEqual(sample[0]['extraction']['paragraphs'][0]['text'], text)
                with self.assertRaises(FileExistsError):
                    prep.build(source, candidate, old, root/'out', count=1)
                source.write_bytes(bz2.compress(b'changed'))
                with self.assertRaisesRegex(ValueError, 'digest'):
                    prep.build(source, candidate, old, root/'bad', count=1)
                self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
