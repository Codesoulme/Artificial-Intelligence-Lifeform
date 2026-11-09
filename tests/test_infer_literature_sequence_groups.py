import unittest

from scripts.infer_literature_sequence_groups import chinese_number, infer_groups


class LiteratureGroupingTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(chinese_number('三十'), 30)
        self.assertEqual(chinese_number('一百零二'), 102)
        self.assertEqual(chinese_number('42'), 42)
        self.assertEqual(chinese_number('二〇二'), 202)

    def test_groups_flag_uncertain_markers_and_missing_chapters(self):
        rows = ['前言', '第一回 开始', '正文', '第三回 继续', '第四回合他输了', '第一章 新标题']
        markers, groups = infer_groups(rows)
        self.assertEqual(len(markers), 4)
        self.assertEqual(groups[0]['start_marker_row'], 1)
        self.assertEqual(groups[0]['end_marker_row'], 4)
        self.assertEqual(groups[0]['row_count'], 4)
        self.assertEqual(groups[0]['characters'], sum(map(len, rows[1:5])))
        self.assertIn('chapter_number_gaps', groups[0]['review_flags'])
        self.assertIn('possible_prose_or_heading_artifacts', groups[0]['review_flags'])
        self.assertIn('single_marker', groups[1]['review_flags'])

    def test_no_markers_has_no_groups(self):
        self.assertEqual(infer_groups(['普通正文']), ([], []))


if __name__ == '__main__':
    unittest.main()
