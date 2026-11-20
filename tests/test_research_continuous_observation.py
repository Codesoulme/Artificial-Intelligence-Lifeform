import unittest

from spora.research_continuous_observation import (
    ADDRESSES, can_register, inherited_query_subject, parse_observation, query_subject, split_history_query,
)


class ContinuousObservationTests(unittest.TestCase):
    def test_history_prefix_keeps_event_time_separate_from_corrections(self):
        question = '沈亦平常最爱喝什么？'
        for prefix, back in (('上一次记录里，', 1), ('请查一下上一次记录：', 1), ('两次更正前，', 2),
                             ('回看3次更正前：', 3), ('十二次更正前 ', 12), ('九十九次更正前，', 99)):
            with self.subTest(prefix=prefix):
                self.assertEqual(split_history_query(prefix + question), (question, back))
        for message in ('沈亦上一次去哪里？', '沈亦何时抵达？', '沈亦两天前去了哪？', question):
            self.assertEqual(split_history_query(message), (message, 0))
        for message in ('上一次记录，', '零次更正前，' + question, '100次更正前，' + question):
            self.assertEqual(split_history_query(message), (message, None))
        self.assertEqual(split_history_query('齐宁上一次记录里最爱喝什么？'), ('齐宁最爱喝什么？', 1))
        self.assertEqual(split_history_query('齐宁两次更正前，职业是什么？'), ('齐宁职业是什么？', 2))
        self.assertEqual(split_history_query('齐宁上一次去哪里？'), ('齐宁上一次去哪里？', 0))
        self.assertEqual(split_history_query('齐宁远上一次记录里最爱喝什么？'),
                         ('齐宁远上一次记录里最爱喝什么？', 0))

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
