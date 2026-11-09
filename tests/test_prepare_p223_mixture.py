import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_p223_mixture as prep
from tests.test_prepare_meow_authored_text import FakeTokenizer


class MixtureTests(unittest.TestCase):
    def test_bound_sources_deduplicate_and_keep_attribution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vocab = root/'vocab'
            vocab.write_bytes(b'vocab')
            sources = []
            for index, name in enumerate(('wiki99', 'wiki100', 'wiki101', 'voyage')):
                directory = root/name
                directory.mkdir()
                text = '中文文本内容' + str(index)
                row = {'text': text, 'text_sha256': prep.sha(text.encode('utf8')), 'title': '标题'+str(index),
                       'page_id': str(index), 'group': 'group'+str(index), 'url': 'https://example.org'}
                raw = (json.dumps(row, ensure_ascii=False)+'\n').encode('utf8')
                filename = 'candidate.jsonl' if name == 'voyage' else 'train.jsonl'
                (directory/filename).write_bytes(raw)
                manifest = {'output_sha256': prep.sha(raw)} if name == 'voyage' else {'files': {'train.jsonl': prep.sha(raw)}}
                manifest_raw = json.dumps(manifest).encode('utf8')
                (directory/'manifest.json').write_bytes(manifest_raw)
                sources.append((name, name, prep.sha(manifest_raw)))
            with patch.object(prep, 'SOURCES', tuple(sources)), \
                 patch.object(prep, 'VOCAB_SHA', prep.sha(vocab.read_bytes())), \
                 patch.object(prep.Tokenizer, 'from_file', return_value=FakeTokenizer()):
                result = prep.build(root, vocab, root/'out')
                rows = [json.loads(line) for line in (root/'out'/'candidate.jsonl').read_text(encoding='utf8').splitlines()]
                self.assertEqual(result['counts']['records'], 4)
                self.assertEqual([r['origin_source'] for r in rows], ['wiki99','wiki100','wiki101','voyage'])
                self.assertTrue(all(r['origin_url'] for r in rows))
                with self.assertRaises(FileExistsError):
                    prep.build(root, vocab, root/'out')
                (root/'wiki99'/'train.jsonl').write_text('{}\n', encoding='utf8')
                with self.assertRaisesRegex(ValueError, 'content changed'):
                    prep.build(root, vocab, root/'changed')


if __name__ == '__main__':
    unittest.main()
