import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_meow_authored_text as prep


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': list(text.encode('utf8'))})()

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode('utf8')


class AuthoredTextTests(unittest.TestCase):
    def test_grouping_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prose, essays, vocab, out = [root / n for n in ('prose', 'essays', 'vocab', 'out')]
            prose.write_text('中文正文。', encoding='utf8')
            rows = [{'id': f'meow-v2-deepdive-{i}', 'title': '标题', 'body': ['第一段。', '第二段。']}
                    for i in range(1, 6)]
            essays.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows), encoding='utf8')
            vocab.write_bytes(b'vocab')
            with patch.object(prep, 'PROSE_SHA', prep.sha(prose.read_bytes())), \
                 patch.object(prep, 'DEEP_SHA', prep.sha(essays.read_bytes())), \
                 patch.object(prep, 'VOCAB_SHA', prep.sha(vocab.read_bytes())), \
                 patch.object(prep.Tokenizer, 'from_file', return_value=FakeTokenizer()):
                result = prep.prepare(prose, essays, vocab, out)
                actual = [json.loads(x) for x in (out/'candidate.jsonl').read_text(encoding='utf8').splitlines()]
                self.assertEqual(result['documents'], 6)
                self.assertEqual({x['work_group'] for x in actual}, {'meow-v2-one-work'})
                self.assertEqual(actual[1]['text'], '第一段。\n第二段。')
                with self.assertRaises(FileExistsError):
                    prep.prepare(prose, essays, vocab, out)
                essays.write_text('{}\n', encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    prep.prepare(prose, essays, vocab, root/'changed')


if __name__ == '__main__':
    unittest.main()
