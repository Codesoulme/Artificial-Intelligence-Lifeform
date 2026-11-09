import bz2
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_wikivoyage_context as prep
from tests.test_prepare_meow_authored_text import FakeTokenizer


class WikivoyageContextTests(unittest.TestCase):
    def test_adjacent_blocks_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, vocab, previous = root/'source.bz2', root/'vocab', root/'previous'
            previous.mkdir()
            a = '这是介绍城市历史与自然景观的完整中文句子。' * 5
            b = '这里还有不同季节的自然风景和文化活动。' * 5
            c = '-{zh-hans:简体;zh-hant:繁體}-' + '这是应当隔离的格式标记。' * 5
            xml = (f'<mediawiki xmlns="urn:test"><page><title>标题</title><ns>0</ns><id>2</id>'
                   f'<revision><id>9</id><text>{a}\n\n{b}\n\n{c}</text></revision></page></mediawiki>')
            source.write_bytes(bz2.compress(xml.encode('utf8')))
            vocab.write_bytes(b'vocab')
            rows = [{'page_id': '2', 'revision_id': '9', 'title': '标题', 'url': 'url', 'group': 'group',
                     'paragraph_ordinal': i, 'text': text, 'text_sha256': prep.sha(text.encode('utf8'))}
                    for i, text in enumerate((a, b, c))]
            raw = ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows).encode('utf8')
            sample = b'[]'
            (previous/'candidate.jsonl').write_bytes(raw)
            (previous/'review-sample.json').write_bytes(sample)
            with patch.object(prep, 'SOURCE_SHA1', prep.digest(source, 'sha1')), \
                 patch.object(prep, 'VOCAB_SHA256', prep.digest(vocab, 'sha256')), \
                 patch.object(prep, 'CANDIDATE_SHA', prep.sha(raw)), \
                 patch.object(prep, 'OLD_SAMPLE_SHA', prep.sha(sample)), \
                 patch.object(prep.Tokenizer, 'from_file', return_value=FakeTokenizer()):
                result = prep.build(source, vocab, previous, root/'out')
                actual = json.loads((root/'out'/'candidate.jsonl').read_text(encoding='utf8'))
                self.assertEqual(result['counts']['spans'], 1)
                self.assertEqual(result['counts']['format_or_time_risk'], 1)
                self.assertEqual(actual['paragraph_ordinals'], [0, 1])
                self.assertEqual(actual['text'], a+'\n\n'+b)
                with self.assertRaises(FileExistsError):
                    prep.build(source, vocab, previous, root/'out')


if __name__ == '__main__':
    unittest.main()
