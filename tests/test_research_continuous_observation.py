import unittest

from spora.research_continuous_observation import ADDRESSES, can_register, inherited_query_subject, parse_observation, query_subject


class ContinuousObservationTests(unittest.TestCase):
    def test_registered_subjects_are_scoped_to_the_supplied_registry(self):
        addresses = {**ADDRESSES, '周岚': 0}
        observation = '周岚最喜欢喝杏仁茶。'
        self.assertIsNone(parse_observation(observation))
        self.assertEqual(parse_observation(observation, addresses), (0, 'drink', '杏仁茶'))
        self.assertEqual(query_subject('周岚平常最爱喝什么？', addresses), (0, 'named_subject'))
        self.assertEqual(query_subject('周岚远平常最爱喝什么？', addresses), (None, 'unresolved_subject'))
        self.assertEqual(query_subject('周岚和何程各自爱喝什么？', addresses), (None, 'ambiguous_subject'))
        self.assertEqual(inherited_query_subject('查到了就说一下。', '周岚的职业有记录。', addresses), 0)
        self.assertTrue(can_register('宋文', addresses))
        self.assertFalse(can_register('周岚', addresses))
        self.assertFalse(can_register('周岚远', addresses))
        self.assertFalse(can_register('Li Lei', addresses))

    def test_query_subject_requires_one_bounded_mention(self):
        for text, expected in (
            ('何程平常最爱喝什么？', (47, 'named_subject')),
            ('想查齐宁当时做的工作。', (26, 'named_subject')),
            ('我翻了何程的资料，想核实他那阵子是靠什么营生的。', (47, 'named_subject')),
            ('何程远平常最爱喝什么？', (None, 'unresolved_subject')),
            ('何诚平常最爱喝什么？', (None, 'no_subject')),
            ('齐宁和何程都喜欢喝什么？', (None, 'ambiguous_subject')),
        ):
            with self.subTest(text=text):
                self.assertEqual(query_subject(text), expected)

    def test_query_subject_inheritance_is_bounded(self):
        history = '齐宁当时的工作有记录。'
        self.assertEqual(inherited_query_subject('查到了就说一下。', history), 26)
        self.assertIsNone(inherited_query_subject('何诚平常最爱喝什么？', history))
        self.assertIsNone(inherited_query_subject('查到了就说一下。', '齐宁和何程当时的工作。'))
        self.assertIsNone(inherited_query_subject('查到了就说一下。', None))

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
