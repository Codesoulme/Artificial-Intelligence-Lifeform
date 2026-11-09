import unittest

from scripts import prepare_p231_wikipedia_context as p


PARA = '这是一段完整的百科正文，描述当地的历史和文化，也介绍不同年代的变化和影响。'
OTHER = '这是一段独立的后续正文，说明当地的地理环境以及居民生活的其他方面和历史背景。'


def fixture(text, body_lines):
    source = {'id': '123', 'title': '真实标题', 'url': 'https://zh.wikipedia.org/wiki/example', 'text': text}
    rows = []
    offset = 0
    for line in body_lines:
        start = text.index(line, offset)
        end = start + len(line)
        offset = end
        rows.append({'status': p.STATUS, 'page_id': source['id'], 'title': source['title'],
                     'url': source['url'], 'source_text_sha256': p.digest(text), 'text': line,
                     'text_sha256': p.digest(line), 'char_start': start, 'char_end': end,
                     'paragraphs': [{'char_start': start, 'char_end': end}]})
    return source, rows


class ContextTests(unittest.TestCase):
    def test_structure_offsets_and_gaps(self):
        raw = '地理\n\n' + PARA + '\n\n' + OTHER + '\n\n* 列表项目\n\n' + PARA + '\n\n（；缺损）\n\n' + OTHER
        source, rows = fixture(raw, [PARA, OTHER, PARA, OTHER])
        result = p.build_page_candidates(source, rows, len)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['text'], PARA + '\n\n' + OTHER)
        self.assertEqual(result[0]['section']['text'], '地理')
        self.assertEqual(raw[result[0]['section']['char_start']:result[0]['section']['char_end']], '地理')
        self.assertIsNone(result[1]['section'])
        for row in result:
            self.assertEqual(raw[row['char_start']:row['char_end']], row['text'])
            self.assertTrue(row['model_text'].startswith('真实标题\n'))
            self.assertEqual(row['extra_heading_bpe_tokens'], len(row['model_text']) - len(row['text']))
            self.assertNotIn('列表项目', row['model_text'])
            self.assertNotIn('缺损', row['model_text'])
        self.assertEqual([p.classify_line(x) for x in ('章节', '* 项目', '「引文」', '（；）')],
                         ['heading', 'list', 'quote', 'damaged'])

    def test_reject_foreign_page_and_forged_spans(self):
        source, rows = fixture('章节\n' + PARA, [PARA])
        foreign = dict(rows[0], page_id='124')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            p.build_page_candidates(source, [foreign], len)
        forged = dict(rows[0], paragraphs=[{'char_start': rows[0]['char_start'] + 1,
                                           'char_end': rows[0]['char_end']}])
        with self.assertRaises(ValueError):
            p.build_page_candidates(source, [forged], len)
        changed = dict(source, text=source['text'] + '其他页面内容')
        with self.assertRaisesRegex(ValueError, 'source mismatch'):
            p.build_page_candidates(changed, rows, len)

    def test_nonadditive_token_counts(self):
        source, rows = fixture('章节\n' + PARA, [PARA])
        def tokens(text):
            return len(text.split('\n'))
        row = p.build_page_candidates(source, rows, tokens)[0]
        self.assertEqual(row['body_bpe_tokens'], 1)
        self.assertEqual(row['extra_heading_bpe_tokens'], 4)
        self.assertEqual(row['model_bpe_tokens'], 4)
        self.assertEqual(row['model_minus_body_bpe_tokens'], 3)

    def test_terminal_section_stops_page(self):
        raw = PARA + '\n参考文献\n' + OTHER
        source, rows = fixture(raw, [PARA, OTHER])
        self.assertEqual([r['text'] for r in p.build_page_candidates(source, rows, len)], [PARA])

    def test_sampling_excludes_both_review_rounds(self):
        rounds = [[str(i) for i in range(40)], [str(i) for i in range(40, 80)]]
        candidates = [{'status': p.STATUS, 'page_id': str(i), 'char_start': 0} for i in range(85)]
        sample = p.sample_candidates(candidates, rounds, 5, 'fixed')
        self.assertEqual({x['page_id'] for x in sample}, {str(i) for i in range(80, 85)})
        self.assertEqual(sample, p.sample_candidates(candidates, rounds, 5, 'fixed'))
        with self.assertRaises(ValueError):
            p.sample_candidates(candidates, [rounds[0], rounds[0]], 1, 'fixed')
        with self.assertRaises(ValueError):
            p.sample_candidates(candidates, rounds, 6, 'fixed')


if __name__ == '__main__':
    unittest.main()
