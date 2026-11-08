import unittest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch
import scripts.verify_storybook_pages as verifier
from scripts.verify_storybook_pages import Paragraphs, normalise


class StoryPageTests(unittest.TestCase):
    def test_extracts_only_story_paragraphs_for_all_reading_levels(self):
        parser = Paragraphs()
        parser.feed('<div><h3>署名</h3></div>'
                    '<div class="column level3-txt def"><h3>第一 <em>段</em>。</h3></div>'
                    '<div class="column level4-txt def"><h3>第二&nbsp;段。</h3></div>'
                    '<div class="column level3-txt translation"><h3>译文</h3></div>')
        self.assertEqual([normalise(p) for p in parser.result], ['第一段。', '第二段。'])

    def test_comparison_preserves_punctuation(self):
        self.assertNotEqual(normalise('你好！'), normalise('你好？'))

    def test_verification_fails_closed_on_hash_or_content_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'raw-v1').mkdir()
            (base / 'license-audit-v1').mkdir()
            source = ('# 标题\n##\n正文。\n##\n* Text: 作者\n'
                      '* Illustration: 绘者\n* Translation: 译者\n* License: [CC-BY]\n').encode()
            page = '<div class="level3-txt def"><h3>正文。</h3></div>'.encode()
            (base / 'raw-v1/0001.md').write_bytes(source)
            (base / 'license-audit-v1/0001.html').write_bytes(page)
            row = {'story': '0001', 'source_label': 'CC-BY',
                   'source_sha256': hashlib.sha256(source).hexdigest(),
                   'page_sha256': hashlib.sha256(page).hexdigest(),
                   'source_path': 'zh/0001.md', 'page_url': 'https://example.com/',
                   'license_urls': ['https://creativecommons.org/licenses/by/3.0/']}
            with patch.object(verifier, 'BASE', base):
                self.assertEqual(verifier.verify(row)['status'], 'matched_pending_quality_and_rights_review')
                (base / 'license-audit-v1/0001.html').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    verifier.verify(row)
                row['page_sha256'] = hashlib.sha256(b'changed').hexdigest()
                self.assertEqual(verifier.verify(row)['status'], 'review_required')


if __name__ == '__main__':
    unittest.main()
