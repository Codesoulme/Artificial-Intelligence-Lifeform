import bz2
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_wikibooks_review as prep
from tests.test_prepare_meow_authored_text import FakeTokenizer


class WikibooksReviewTests(unittest.TestCase):
    def test_excludes_prior_work_and_preserves_page_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, vocab, audit = root/'dump.bz2', root/'vocab', root/'audit.json'
            titles = [f'教材{i}' for i in range(100)
                      if int(prep.sha(f'教材{i}'.encode())[:8], 16) % 10 != 0][:2]
            text = '这是一段中文教材内容，用于检查页面来源和抽取结果。' * 20
            pages = ''.join(f'<page><title>{title}/第一章</title><ns>0</ns><id>{i}</id>'
                            f'<revision><id>{i+10}</id><text>{text}</text></revision></page>'
                            for i, title in enumerate(titles, 1))
            source.write_bytes(bz2.compress(f'<mediawiki xmlns="urn:test">{pages}</mediawiki>'.encode()))
            vocab.write_bytes(b'vocab')
            source_sha = hashlib.sha1(source.read_bytes()).hexdigest()
            audit.write_text(json.dumps({'source_sha1': source_sha, 'sample_size': 1,
                                         'samples': [{'work_sha256': prep.sha(titles[0].encode())}]}), encoding='utf8')
            with patch.object(prep, 'EXPECTED_SHA1', source_sha), \
                 patch.object(prep, 'VOCAB_SHA256', prep.sha(vocab.read_bytes())), \
                 patch.object(prep, 'AUDIT_SHA256', prep.sha(audit.read_bytes())), \
                 patch.object(prep.Tokenizer, 'from_file', return_value=FakeTokenizer()):
                one = prep.build(source, vocab, audit, root/'one')
                two = prep.build(source, vocab, audit, root/'two')
                self.assertEqual(one, two)
                self.assertEqual(one['counts']['pages'], 1)
                row = json.loads((root/'one'/'candidate.jsonl').read_text(encoding='utf8'))
                self.assertEqual((row['page_id'], row['revision_id']), ('2', '12'))
                self.assertEqual(row['title'], titles[1]+'/第一章')
                self.assertEqual(row['text_sha256'], prep.sha(row['text'].encode()))
                sample = json.loads((root/'one'/'review-sample.json').read_text(encoding='utf8'))
                self.assertEqual(sample[0]['raw_sha256'], prep.sha(sample[0]['raw_wikitext'].encode()))
                with self.assertRaises(FileExistsError):
                    prep.build(source, vocab, audit, root/'one')
                vocab.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'digest'):
                    prep.build(source, vocab, audit, root/'three')
                self.assertFalse((root/'three').exists())


if __name__ == '__main__':
    unittest.main()
