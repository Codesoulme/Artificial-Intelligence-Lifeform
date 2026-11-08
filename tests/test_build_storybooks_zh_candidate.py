import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_storybooks_zh_candidate as build

class CandidateTests(unittest.TestCase):
    def test_paragraphs_and_attribution_preserve_internal_lines(self):
        raw = '# 标题\n\n##\n第一行\n第二行\n\n##\n末段。\n\n##\n* License: [CC-BY]\n* Text: A\n* Illustration: B\n* Translation: C\n* Language: zh\n'
        title, paragraphs, attrs = build.extract(raw, 2)
        self.assertEqual((title, paragraphs), ('标题', ['第一行\n第二行', '末段。']))
        self.assertEqual(attrs['Translation'], 'C')
        with self.assertRaises(ValueError): build.extract(raw, 1)
        with self.assertRaises(ValueError): build.extract(raw.replace('[CC-BY]', '[CC-BY-NC]'), 2)
    def test_page_paragraph_order_and_license(self):
        p = build.Page()
        p.feed('<a rel="license" href="https://creativecommons.org/licenses/by/4.0/">L</a><div class="level1-txt def"><h3>一<b>二</b></h3><h3>三</h3></div>')
        self.assertEqual(p.paragraphs, ['一二','三'])
        self.assertEqual(p.licenses, {'https://creativecommons.org/licenses/by/4.0/'})
    def test_hash_split_is_stable_and_story_based(self):
        a = build.split_for('0001')
        self.assertEqual(a, build.split_for('0001'))
        self.assertEqual(a[0], 'dev' if int(a[1],16)%10 == 0 else 'train')
        self.assertNotEqual(a[1], build.split_for('0002')[1])
    def test_overlap_detects_complete_and_window_matches(self):
        texts = ['abcdefghijklmnopqrstuv', '0abcdefghijklmnopqrstuv9', '完全不同的文本内容']
        got = build.overlaps(texts)
        self.assertEqual(got[0]['full_containment_peers'], 1)
        self.assertEqual(got[1]['full_containment_peers'], 0)
        self.assertEqual(got[0]['20_peers'], 1)
        self.assertEqual(got[2]['12_peers'], 0)
    def test_duplicate_entries_rejected(self):
        with self.assertRaises(ValueError): build.keyed([{'story':'a'}, {'story':'a'}])
    def test_exclusive_directory_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = build.output_path('candidate', Path(tmp))
            p.mkdir(exist_ok=False)
            with self.assertRaises(FileExistsError): p.mkdir(exist_ok=False)
            (p / 'sentinel').write_bytes(b'unchanged')
            with patch.object(build, 'output_path', return_value=p), patch('sys.argv', ['build', '--output-name', 'candidate']):
                with self.assertRaises(FileExistsError): build.main()
            self.assertEqual((p / 'sentinel').read_bytes(), b'unchanged')

    def test_output_path_rejects_nested_and_absolute_names(self):
        for name in ('', '.', '..', '../other', 'a/b', 'a\\b', 'C:other'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build.output_path(name)

if __name__ == '__main__': unittest.main()
