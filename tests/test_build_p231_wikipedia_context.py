import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import build_p231_wikipedia_context as build
from scripts import prepare_p231_wikipedia_context as context


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return build.sha(raw)


def encoded(value):
    return json.dumps(value, ensure_ascii=False).encode('utf-8')


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': list(text)})()


class BuildTests(unittest.TestCase):
    def fixture(self, root):
        sources, rows = [], []
        for i in range(85):
            # The last page duplicates another eligible page's body.
            number = 80 if i == 84 else i
            text = '章节\n这是一段完整的百科正文，描述当地的历史和文化，也介绍不同年代的变化和影响。编号' + str(number) + '。'
            start = len('章节\n')
            pid, title = str(i + 100), '标题' + str(i)
            source = {'id': pid, 'title': title, 'url': 'https://zh.wikipedia.org/wiki/' + pid, 'text': text}
            sources.append(source)
            body = text[start:]
            rows.append({'status': context.STATUS, 'page_id': pid, 'title': title, 'url': source['url'],
                         'source_text_sha256': context.digest(text), 'text': body,
                         'text_sha256': context.digest(body), 'char_start': start, 'char_end': len(text),
                         'paragraphs': [{'char_start': start, 'char_end': len(text)}], 'bpe_tokens': len(body)})
        sources.append({'id': '999', 'title': '其他页面', 'url': 'https://example.org', 'text': '不得输出'})
        source_path = root / build.SOURCE
        source_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(sources), source_path, row_group_size=10)
        token_hash = write(root / build.TOKENIZER, b'synthetic tokenizer')
        source_hash = build.sha(source_path.read_bytes())
        sample1, sample2 = rows[:40], rows[40:80]
        candidate_hash = write(root / build.REFINED / 'candidates.jsonl',
                               b''.join(encoded(row) + b'\n' for row in rows))
        sample2_hash = write(root / build.REFINED / 'sample.json', encoded(sample2))
        sample1_hash = write(root / build.FIRST / 'sample.json', encoded(sample1))
        first_manifest = {'status': 'training_side_sample_pending_review', 'candidate_manifest_sha256': 'parent',
                          'candidate_sha256': 'parent_rows', 'sample_sha256': sample1_hash}
        refined = {'status': context.STATUS, 'source_manifest_sha256': 'parent',
                   'source_files_sha256': {'candidates.jsonl': 'parent_rows'},
                   'files': {'candidates.jsonl': candidate_hash, 'sample.json': sample2_hash},
                   'counts': {'retained_passages': len(rows)}}
        pins = {'refined_manifest': write(root / build.REFINED / 'manifest.json', encoded(refined)),
                'first_manifest': write(root / build.FIRST / 'manifest.json', encoded(first_manifest)),
                'source': source_hash, 'tokenizer': token_hash, 'reviews': {}}
        for prefix, sample in [('p230-body-review-', sample1), ('p230-refined-review-', sample2)]:
            for start in (0, 10, 20, 30):
                name = prefix + str(start) + '-' + str(start + 9) + '.json'
                reviews = [{'index': i, 'page_id': sample[i]['page_id'],
                            'text_sha256': sample[i]['text_sha256']} for i in range(start, start + 10)]
                pins['reviews'][name] = write(root / 'runs' / name, encoded(reviews))
        return pins

    def test_build_provenance_grouping_dedup_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pins = self.fixture(root)
            out = root / 'runs' / 'result'
            with patch.object(build.Tokenizer, 'from_file', return_value=FakeTokenizer()):
                manifest = build.build(root, out, pins, sample_count=4)
            self.assertEqual(manifest['counts']['exact_body_duplicates_removed'], 1)
            self.assertEqual(manifest['counts']['candidate_pages'], 84)
            self.assertEqual(manifest['counts']['full_rows_decoded'], 86)
            sample = json.loads((out / 'sample.json').read_text(encoding='utf-8'))
            self.assertEqual({r['page_id'] for r in sample}, {'180', '181', '182', '183'})
            pages = [json.loads(line) for line in (out / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertNotIn('999', {p['page_id'] for p in pages})
            row = pages[0]['candidates'][0]
            self.assertIn(row['title'], row['model_text'])
            self.assertEqual(row['section']['text'], '章节')
            self.assertEqual(row['model_bpe_tokens'], len(row['model_text']))
            self.assertEqual(len((out / 'duplicates.jsonl').read_text(encoding='utf-8').splitlines()), 1)
            with self.assertRaises(ValueError):
                build.build(root, out, pins, sample_count=4)

    def test_tampering_fails_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pins = self.fixture(root)
            (root / build.REFINED / 'sample.json').write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Digest mismatch'):
                build.build(root, root / 'runs' / 'out', pins, sample_count=4)
            self.assertFalse((root / 'runs' / 'out').exists())

    def test_mismatched_source_and_duplicate_retained_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pins = self.fixture(root)
            groups, _, _, _ = build.load_inputs(root, pins)
            groups['100'].append(groups['100'][0])
            with self.assertRaisesRegex(ValueError, 'paragraph spans'):
                build.construct(groups, root / build.SOURCE, len)
            groups['100'].pop()
            groups['100'][0]['source_text_sha256'] = 'bad'
            with self.assertRaisesRegex(ValueError, 'source mismatch'):
                build.construct(groups, root / build.SOURCE, len)


if __name__ == '__main__':
    unittest.main()
