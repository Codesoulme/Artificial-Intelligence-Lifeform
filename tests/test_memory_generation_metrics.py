import unittest

from scripts.spora_memory_generation_metrics import asserted_fact, score_pair


class GenerationMetricsTests(unittest.TestCase):
    def test_lists_are_not_correct_assertions(self):
        self.assertFalse(asserted_fact('小赵喜欢的水果包括苹果、香蕉。', '小赵', '苹果'))

    def test_natural_fact_rephrasings_are_correct(self):
        self.assertTrue(asserted_fact('小赵爱吃的水果是苹果。', '小赵', '苹果'))
        self.assertTrue(asserted_fact('小赵水果偏好是苹果。', '小赵', '苹果'))

    def test_denial_and_wrong_subject_do_not_pass(self):
        self.assertFalse(asserted_fact('小赵不喜欢苹果。', '小赵', '苹果'))
        self.assertFalse(asserted_fact('小林喜欢的水果是苹果。', '小赵', '苹果'))

    def test_changed_but_wrong_answer_does_not_pass_swap(self):
        result = score_pair('苹果', '西瓜', name='小赵', original='苹果', replacement='香蕉')
        self.assertFalse(result['original_still_present'])
        self.assertFalse(result['pair_correct'])

    def test_both_counterfactual_worlds_must_be_correct(self):
        result = score_pair('小赵喜欢的水果是苹果。', '香蕉',
                            name='小赵', original='苹果', replacement='香蕉')
        self.assertTrue(result['pair_correct'])
        self.assertFalse(score_pair('西瓜', '香蕉', name='小赵', original='苹果',
                                    replacement='香蕉')['pair_correct'])

    def test_complete_multitoken_fact_is_required(self):
        self.assertFalse(asserted_fact('火龙', '小赵', '火龙果'))
        self.assertTrue(asserted_fact('小赵喜欢的水果是火龙果。', '小赵', '火龙果'))


if __name__ == '__main__':
    unittest.main()
