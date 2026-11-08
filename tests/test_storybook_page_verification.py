import unittest
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


if __name__ == '__main__':
    unittest.main()
