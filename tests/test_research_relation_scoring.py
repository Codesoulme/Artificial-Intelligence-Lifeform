import unittest

from spora.research_relation_scoring import score_relation_answer


class RelationScoringTests(unittest.TestCase):
    def check(self, answer, expected, subject="小林", relation="饮品", value="酸奶"):
        self.assertIs(score_relation_answer(answer, subject, relation, value), expected, answer)

    def test_complete_positive_templates_and_names(self):
        for answer in ("小林喜欢喝酸奶", "小林喜欢喝的是酸奶。", "小林喜欢喝的饮料是酸奶。", "小林喜欢的饮品是酸奶。",
                       "  小林喜欢的饮料是酸奶.  ", "小林喜欢喝 酸奶。"):
            with self.subTest(answer=answer):
                self.check(answer, True)
        self.check("欧阳娜娜喜欢的颜色是蓝色。", True, "欧阳娜娜", "颜色", "蓝色")
        self.check("阿依·古丽喜欢的活动是跑步！", True, "阿依·古丽", "活动", "跑步")

    def test_wrong_subject_and_relation(self):
        for answer in ("小梅喜欢喝酸奶。", "小林喜欢的颜色是酸奶。",
                       "小林喜欢的活动是酸奶。", "小林喜欢酸奶。"):
            with self.subTest(answer=answer):
                self.check(answer, False)
        self.check("小林喜欢的颜色是蓝色。", False, relation="活动", value="蓝色")

    def test_conflicting_values_and_subjects(self):
        for answer in ("小林喜欢喝酸奶和牛奶。", "小林喜欢喝牛奶，酸奶。",
                       "小林喜欢喝酸奶，也喜欢喝牛奶。",
                       "小林和小梅喜欢喝酸奶。", "小林喜欢喝酸奶，小梅喜欢喝茶。",
                       "小梅喜欢喝酸奶，小林喜欢喝牛奶。"):
            with self.subTest(answer=answer):
                self.check(answer, False)

    def test_negation_uncertainty_and_partial_answers(self):
        for answer in ("", "  ", "酸奶", "小林酸奶", "小林不喜欢喝酸奶。",
                       "小林喜欢的饮品不是酸奶。", "小林可能喜欢喝酸奶。",
                       "小林喜欢喝酸奶吗？", "小林喜欢喝酸奶。也许。",
                       "小林喜欢喝酸奶。。"):
            with self.subTest(answer=answer):
                self.check(answer, False)

    def test_unsupported_or_unsafe_inputs_fail_closed(self):
        for subject, relation, value in (("", "饮品", "酸奶"),
                                         ("小林和小梅", "饮品", "酸奶"),
                                         ("小林", "饮品", "酸奶和牛奶"),
                                         ("小林", "动物", "酸奶"),
                                         ("小林", "饮品", "")):
            with self.subTest(subject=subject, relation=relation, value=value):
                self.check("小林喜欢喝酸奶。", False, subject, relation, value)
        self.assertFalse(score_relation_answer(None, "小林", "饮品", "酸奶"))

    def test_complete_value_not_prefix_or_substring(self):
        self.check("小林喜欢喝酸奶饮料。", False)
        self.check("小林喜欢喝酸奶。", False, value="奶")
        self.check("小林喜欢喝酸奶。", False, subject="林")


if __name__ == "__main__":
    unittest.main()
