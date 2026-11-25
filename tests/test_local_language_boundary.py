import unittest

from scripts.audit_spora_local_language_boundary import HELD, SMOKE_SCORING, assess


class LocalLanguageBoundaryTests(unittest.TestCase):
    def test_words_in_a_forbidden_pattern_are_not_individual_characters(self):
        case = next(row for row in HELD if row[0] == 'update-invitation')
        required, forbidden, max_chars = case[-3:]
        revised = '小孟，周日上午学校东门见，记得带杯子。'
        self.assertTrue(assess(revised, True, required, forbidden, max_chars)['pass'])
        self.assertFalse(assess(revised + '周六下午也可以。', True, required, forbidden, max_chars)['pass'])

    def test_evaluator_rejects_a_bare_pattern_instead_of_iterating_characters(self):
        with self.assertRaises(TypeError):
            assess('请关窗。', True, (r'关窗',), r'灯|随后', 15)

    def test_every_case_supplies_pattern_collections(self):
        for required, forbidden, limit in (*SMOKE_SCORING.values(), *(case[-3:] for case in HELD)):
            self.assertIsInstance(required, tuple)
            self.assertIsInstance(forbidden, tuple)
            self.assertGreater(limit, 0)

    def test_one_step_recognizes_an_action_with_natural_wording(self):
        required, forbidden, limit = SMOKE_SCORING['one-step']
        self.assertTrue(assess('把书桌里的废纸拿出来。', True, required, forbidden, limit)['pass'])
        self.assertFalse(assess('先把桌子里的废纸。', True, required, forbidden, limit)['pass'])
