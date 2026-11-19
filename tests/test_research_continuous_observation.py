import unittest

from spora.research_continuous_observation import parse_observation


class ContinuousObservationTests(unittest.TestCase):
    def test_extracts_exact_subject_field_and_value(self):
        for text, expected in (
            ('何程现在最喜欢喝青梅花茶。', (47, 'drink', '青梅花茶')),
            (' 齐宁的职业改为船舶检验员。 ', (26, 'work', '船舶检验员')),
            ('何程那次到访了松桥书院。', (47, 'place', '松桥书院')),
            ('齐宁的饮料改成茶', (26, 'drink', '茶')),
            ('何程的目的地是紫杉峡谷.', (47, 'place', '紫杉峡谷')),
            ('齐宁从事的工作是磨具技师', (26, 'work', '磨具技师')),
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_observation(text), expected)

    def test_rejects_unsupported_and_ambiguous_statements(self):
        for text in (
            '何程是谁？', '何程最喜欢喝清茶吗', '何程的饮料是咖啡？', '何程不喜欢喝咖啡。',
            '听说何程的工作是导游。', '何程的职业是可能是导游。', '何程的职业是导游才怪。',
            '何程的职业是导游或者画家。', '何程的职业是导游也是画家吗', '何程的职业是导游，齐宁也是。',
            '何程的职业是导游同时去了码头。', '何程的职业是齐宁的同事。', '何诚的职业是导游。',
            '她的目的地是码头。', '齐宁去那里是上午。', '何程的饮料是。', '何程的饮料是水!',
        ):
            with self.subTest(text=text):
                self.assertIsNone(parse_observation(text))
