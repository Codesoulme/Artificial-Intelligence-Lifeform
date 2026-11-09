import bz2
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_wikivoyage_paragraphs as prep
from tests.test_prepare_meow_authored_text import FakeTokenizer


class WikivoyageTests(unittest.TestCase):
    def test_paragraph_filters_markup_and_contacts(self):
        text = '这是一段介绍当地风景和历史的完整中文内容。' * 6
        self.assertEqual(prep.extract_paragraph(text), text)
        for prefix in ('{{模板}}', '<ref>资料</ref>', '电话123456789', '* 列表', 'https://example.org'):
            self.assertIsNone(prep.extract_paragraph(prefix + text))
        linked = '这里可以看到[[历史建筑|古老建筑]]，' + text
        self.assertIn('古老建筑', prep.extract_paragraph(linked))

    def test_deterministic_page_bucket_and_source_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, vocab = root/'source.bz2', root/'tokenizer'
            paragraph = '这是用于验证页面保留分组的完整中文段落。' * 6
            available = next(str(i) for i in range(1, 100) if int(hashlib.sha256((prep.SALT+str(i)).encode()).hexdigest()[:8],16)%10 != 0)
            reserved = next(str(i) for i in range(1, 100) if int(hashlib.sha256((prep.SALT+str(i)).encode()).hexdigest()[:8],16)%10 == 0)
            xml = ('<mediawiki xmlns="urn:test">' + ''.join(
                f'<page><title>标题{i}</title><ns>0</ns><id>{i}</id><revision><id>8</id><text>{paragraph}</text></revision></page>'
                for i in (available, reserved)) + '</mediawiki>')
            source.write_bytes(bz2.compress(xml.encode('utf8')))
            vocab.write_bytes(b'vocab')
            with patch.object(prep, 'SOURCE_SHA1', prep.digest(source, 'sha1')), \
                 patch.object(prep, 'VOCAB_SHA256', prep.digest(vocab, 'sha256')), \
                 patch.object(prep.Tokenizer, 'from_file', return_value=FakeTokenizer()):
                result = prep.prepare(source, vocab, root/'out')
                rows = [json.loads(x) for x in (root/'out'/'candidate.jsonl').read_text(encoding='utf8').splitlines()]
                self.assertEqual([r['page_id'] for r in rows], [available])
                self.assertEqual(result['counts']['reserved_pages'], 1)
                with self.assertRaises(FileExistsError):
                    prep.prepare(source, vocab, root/'out')
                source.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    prep.prepare(source, vocab, root/'other')


if __name__ == '__main__':
    unittest.main()
