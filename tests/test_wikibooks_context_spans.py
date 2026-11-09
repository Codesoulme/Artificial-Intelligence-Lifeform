import copy
import unittest

from scripts.wikibooks_context_spans import build_spans
from scripts.wikibooks_structured import extract


class WikibooksContextSpanTests(unittest.TestCase):
    def test_consecutive_paragraphs_preserve_offsets_gaps_and_links(self):
        raw = ' \n第一段 [[目标|文字]]。\r\n \t\r\n\r\n第二段。\n\n第三段。  '
        extraction = extract(raw)
        result = build_spans(raw, extraction)
        self.assertEqual(result['status'], 'review_only_not_approved_for_training')
        self.assertEqual(result['offset_unit'], 'unicode_codepoint')
        span, = result['spans']
        self.assertEqual(span['raw'], raw[span['start']:span['end']])
        self.assertEqual(span['text'], ' \n第一段 文字。\r\n \t\r\n\r\n第二段。\n\n第三段。  ')
        self.assertEqual(span['paragraphs'], extraction['paragraphs'])
        link, = span['paragraphs'][0]['links']
        self.assertEqual(raw[link['start']:link['end']], link['raw'])
        self.assertEqual(link['target'], '目标')

    def test_quarantine_and_heading_always_break_spans(self):
        raw = '前段\n\n{{模板}}\n\n中段\n\n== 标题 ==\n正文\n\n后段'
        spans = build_spans(raw, extract(raw))['spans']
        self.assertEqual([span['text'] for span in spans], ['前段', '中段', '后段'])
        for span in spans:
            self.assertEqual(raw[span['start']:span['end']], span['raw'])

    def test_empty_and_all_quarantined(self):
        for raw in ('', ' \n\n ', '{{模板}}', '== 标题 ==\n正文'):
            with self.subTest(raw=raw):
                self.assertEqual(build_spans(raw, extract(raw))['spans'], [])

    def test_uncertain_boundary_never_rejoins_later_text(self):
        raw = '安全\n\n{{未结束\n\n后文'
        self.assertEqual([s['text'] for s in build_spans(raw, extract(raw))['spans']], ['安全'])

    def test_rejects_missing_or_inconsistent_source(self):
        raw = '甲\n\n乙\n\n丙'
        extraction = extract(raw)
        missing = copy.deepcopy(extraction)
        del missing['paragraphs'][1]
        altered = copy.deepcopy(extraction)
        altered['paragraphs'][0]['raw'] = '丙'
        unsafe = copy.deepcopy(extraction)
        unsafe['paragraphs'][0]['risks'] = [{'reason': 'unknown'}]
        changed_text = copy.deepcopy(extraction)
        changed_text['paragraphs'][0]['text'] = '被修改的内容'
        for invalid in (missing, altered, unsafe, changed_text):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    build_spans(raw, invalid)
        with self.assertRaises(TypeError):
            build_spans(None, extraction)


if __name__ == '__main__':
    unittest.main()
