import unittest

from scripts.wikibooks_structured import extract


class WikibooksStructuredTests(unittest.TestCase):
    def test_paragraphs_and_links_preserve_text_and_offsets(self):
        raw = '第一行 [[教材|教科書]]。\n下一行 [[文字]]。\n\n第二段。'
        result = extract(raw)
        self.assertEqual(result['status'], 'review_only_not_approved_for_training')
        first, second = result['paragraphs']
        self.assertEqual(first['text'], '第一行 教科書。\n下一行 文字。')
        self.assertEqual(second['text'], '第二段。')
        self.assertEqual([(link['target'], link['text']) for link in first['links']],
                         [('教材', '教科書'), ('文字', '文字')])
        for item in (first, second, *first['links']):
            self.assertEqual(raw[item['start']:item['end']], item['raw'])

    def test_unsafe_nodes_quarantine_whole_paragraph_without_joining(self):
        cases = [('{{术语|内容}}', 'template'), ('[[File:图.png|说明]]', 'image'),
                 ('<math>x+y</math>', 'formula'), ('<syntaxhighlight>print(1)</syntaxhighlight>', 'code'),
                 ('<ref>来源</ref>', 'citation'), ('[https://example.org 来源]', 'external_link'),
                 ('[[Category:教材]]', 'ambiguous_link'), ('[[词条|]]', 'ambiguous_link'),
                 ('<!-- 隐藏 -->', 'comment'), ('&nbsp;', 'entity'),
                 ('-{zh:文字}-', 'unparsed_markup')]
        for markup, reason in cases:
            with self.subTest(markup=markup):
                raw = '前文' + markup + '后文\n\n完整段落。'
                first, second = extract(raw)['paragraphs']
                self.assertEqual(first['status'], 'quarantined')
                self.assertIsNone(first['text'])
                self.assertIn(reason, [risk['reason'] for risk in first['risks']])
                for risk in first['risks']:
                    self.assertEqual(raw[risk['start']:risk['end']], risk['raw'])
                if reason == 'unparsed_markup':
                    self.assertIsNone(second['text'])
                    self.assertEqual(second['risks'][0]['reason'], 'uncertain_prior_markup_boundary')
                else:
                    self.assertEqual(second['text'], '完整段落。')

    def test_nested_blank_lines_remain_with_unsafe_node(self):
        raw = '前{{模板|一\n\n二}}后\n\n安全段'
        first, second = extract(raw)['paragraphs']
        self.assertEqual(first['raw'], '前{{模板|一\n\n二}}后')
        self.assertEqual(first['risks'][0]['raw'], '{{模板|一\n\n二}}')
        self.assertEqual(second['text'], '安全段')

    def test_unparsed_markup_makes_later_boundaries_uncertain(self):
        raw = '前{{未结束\n\n后文\n\n另一段'
        paragraphs = extract(raw)['paragraphs']
        self.assertEqual(len(paragraphs), 3)
        self.assertEqual(paragraphs[0]['risks'][0]['reason'], 'unparsed_markup')
        for paragraph in paragraphs[1:]:
            self.assertIsNone(paragraph['text'])
            self.assertEqual(paragraph['risks'][0]['reason'], 'uncertain_prior_markup_boundary')
            self.assertEqual(raw[paragraph['start']:paragraph['end']], paragraph['risks'][0]['raw'])

    def test_structure_and_nested_link_are_not_silently_converted(self):
        for markup in ('== 标题 ==\n正文', '* 条目', '{|\n| 表格\n|}',
                       '[[目标|{{模板}}]]', '[[目标|甲|乙]]', 'https://example.org'):
            with self.subTest(markup=markup):
                paragraph = extract(markup)['paragraphs'][0]
                self.assertEqual(paragraph['status'], 'quarantined')
                self.assertTrue(paragraph['risks'])

    def test_empty_and_input_type(self):
        self.assertEqual(extract(' \n\n')['paragraphs'], [])
        with self.assertRaises(TypeError):
            extract(None)


if __name__ == '__main__':
    unittest.main()
