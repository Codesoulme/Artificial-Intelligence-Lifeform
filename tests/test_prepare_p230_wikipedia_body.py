import collections
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_p230_wikipedia_body as p


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class PrepareBodyTests(unittest.TestCase):
    def test_selection_digest_and_train_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = '一个足够明确的百科条目。'
            while int(p.text_sha(p.normalized(text))[:8], 16) % 20 == 0:
                text += '内容'
            row = {'page_id': '123', 'title': '条目', 'url': 'https://zh.wikipedia.org/wiki/example',
                   'text': text, 'group': p.text_sha(p.normalized(text)), 'prior_review_group': False}
            screened = root / 'screened.jsonl'
            screened.write_text(json.dumps(row, ensure_ascii=False) + '\n' + json.dumps({**row, 'page_id': '124', 'prior_review_group': True}) + '\n', encoding='utf-8')
            manifest = root / 'manifest.json'
            write_json(manifest, {'status': 'mechanical_screen_only_not_approved', 'files': {'screened.jsonl': p.sha(screened)},
                                  'counts': {'screened': 2}, 'eligible_counts': {'screened': 1}})
            expected = {'screen': p.sha(screened), 'manifest': p.sha(manifest)}
            self.assertEqual(list(p.load_selection(screened, manifest, expected)), ['123'])
            screened.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                p.load_selection(screened, manifest, expected)
            manifest.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                p.load_selection(screened, manifest, expected)

    def test_exclusion_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'input'
            base.mkdir()
            data = base / 'train.jsonl'
            data.write_text('{"page_id":"123"}\n', encoding='utf-8')
            manifest = base / 'manifest.json'
            write_json(manifest, {'files': {'train.jsonl': p.sha(data)}})
            spec = {'reviewed': ('input', p.sha(manifest), 'train.jsonl')}
            excluded, provenance = p.load_exclusions(root, spec)
            self.assertEqual(excluded, {'123': {'reviewed'}})
            self.assertEqual(provenance['reviewed']['page_count'], 1)
            data.write_text('{"page_id":"124"}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                p.load_exclusions(root, spec)

    def test_queue_coverage_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'queue'
            base.mkdir()
            selected, files = {}, {}
            for n in range(1, 5):
                pid = str(n)
                text = '示例条目正文。' + pid
                source = {'group': 'group' + pid, 'title': '标题' + pid,
                          'url': 'https://zh.wikipedia.org/wiki/' + pid, 'text': text}
                selected[pid] = source
                row = {'page_id': pid, 'group': source['group'], 'title': source['title'],
                       'source_url': source['url'], 'text': text, 'text_sha256': p.text_sha(text)}
                shard = {'status': 'pending_not_approved_for_training', 'source_sha256': p.EXPECTED['source'],
                         'train_sha256': 'train', 'screened_sha256': p.EXPECTED['screen'], 'samples': [row]}
                name = f'review-{n:02}.json'
                write_json(base / name, shard)
                files[name] = p.sha(base / name)
            manifest = {'status': 'pending_not_approved_for_training', 'raw_source_sha256': p.EXPECTED['source'],
                        'train_v3_sha256': 'train', 'p77_manifest_sha256': p.EXPECTED['manifest'],
                        'p77_screened_sha256': p.EXPECTED['screen'], 'files': files,
                        'counts': {'selected': 4, 'shards': 4, 'per_shard': 1}}
            write_json(base / 'manifest.json', manifest)
            spec = {'test_queue': ('queue', p.sha(base / 'manifest.json'), 4)}
            queues, _, provenance = p.load_queues(selected, root, spec)
            self.assertEqual(queues['test_queue'], set(selected))
            self.assertEqual(provenance['test_queue']['files'], files)
            selected['1']['title'] = '不一致'
            with self.assertRaisesRegex(ValueError, 'Queue page mismatch'):
                p.load_queues(selected, root, spec)
            selected['1']['title'] = '标题1'
            (base / 'review-01.json').write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                p.load_queues(selected, root, spec)

    def test_queue_manifest_requires_all_four_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'queue'
            base.mkdir()
            manifest = {'status': 'pending_not_approved_for_training', 'raw_source_sha256': p.EXPECTED['source'],
                        'p77_manifest_sha256': p.EXPECTED['manifest'], 'p77_screened_sha256': p.EXPECTED['screen'],
                        'files': {}, 'counts': {'selected': 4, 'shards': 4, 'per_shard': 1}}
            write_json(base / 'manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'Queue manifest mismatch'):
                p.load_queues({}, root, {'test_queue': ('queue', p.sha(base / 'manifest.json'), 4)})

    def test_candidate_link_requires_queue_membership_and_pinned_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'candidate'
            base.mkdir()
            builder = root / 'builder.py'
            builder.write_text('# builder', encoding='utf-8')
            train = base / 'train.jsonl'
            train.write_text('{"page_id":"1"}\n', encoding='utf-8')
            manifest = {'status': 'candidate_not_approved_for_training', 'source_sha256': p.EXPECTED['source'],
                        'builder_sha256': p.sha(builder), 'files': {'train.jsonl': p.sha(train)},
                        'counts': {'train': {'documents': 1}, 'development': {'documents': 1}},
                        'input_sha256': {'manifest.json': 'queuehash', 'review-01.json': 'shardhash'}}
            write_json(base / 'manifest.json', manifest)
            exclusions = {'candidate': ('candidate', p.sha(base / 'manifest.json'), 'train.jsonl')}
            queues = {'queue': ('unused', 'queuehash', 2)}
            links = (('candidate', 'queue', 'builder.py', ()),)
            with patch.object(p, 'EXCLUSIONS', exclusions), patch.object(p, 'QUEUES', queues), patch.object(p, 'CANDIDATE_LINKS', links):
                source = {'queue': {'files': {'review-01.json': 'shardhash'}}}
                p.verify_candidate_links({'queue': {'1', '2'}}, source, root)
                with self.assertRaisesRegex(ValueError, 'train coverage mismatch'):
                    p.verify_candidate_links({'queue': {'2', '3'}}, source, root)
                builder.write_text('# changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Checksum'):
                    p.verify_candidate_links({'queue': {'1', '2'}}, source, root)

    def test_offsets_and_removals(self):
        a = '这是一段完整的百科正文，描述当地的历史和文化，也介绍不同年代的变化和影响。'
        b = '这是一段独立的后续正文，说明当地的地理环境以及居民生活的其他方面和历史背景。'
        raw = a + '\n\n联系邮箱 example@example.com 不应被保留，这是一段需要剔除的文本内容。\n\n' + b + '\n\n参考文献\n' + a
        stats = collections.Counter()
        spans = list(p.extract(raw, stats))
        self.assertEqual([raw[x[0][0]:x[-1][1]] for x in spans], [a, b])
        self.assertEqual(stats['rejected_format_or_sensitive'], 1)
        self.assertEqual(stats['terminal_sections'], 1)

    def test_oversize_dependent_and_math(self):
        a = '有关这个地区的文字内容相当完整，介绍历史发展和社会生活的多个不同方面。'
        raw = a + '\n\n' + '这是一段超长文字' * 1000 + '。\n\n此外，' + a + '\n\n' + a
        stats = collections.Counter()
        spans = list(p.extract(raw, stats))
        self.assertEqual([raw[x[0][0]:x[-1][1]] for x in spans], [a, a])
        self.assertEqual(stats['rejected_oversize'], 1)
        self.assertEqual(stats['rejected_dependent_opening'], 1)
        self.assertEqual(p.rejection('残缺的数学表达式 \\frac{x}{y}。' * 4), 'format_or_sensitive')

    def test_chunk_boundary_never_splits_paragraph(self):
        para = '有关这个地区的文字内容相当完整，介绍历史发展和社会生活的多个不同方面。'
        raw = para + '\n\n' + para
        with patch.object(p, 'MAX_CHARS', len(para) + 10):
            spans = list(p.extract(raw, collections.Counter()))
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(raw[x[0][0]:x[-1][1]] == para for x in spans))

    def test_output_boundaries(self):
        for name in ['', '.', '..', '../escape', 'a/b', 'a\\b', 'D:escape']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                p.output_path(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'runs' / 'exists'
            out.mkdir(parents=True)
            with patch.object(p, 'ROOT', root), self.assertRaises(FileExistsError):
                p.build(out)


if __name__ == '__main__':
    unittest.main()
