import json
import unittest

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

if transformers is not None:
    from scripts.train_spora_dialogue_composition import DATA, parse_answer, request_tiles


@unittest.skipIf(transformers is None, 'requires optional open-base dependencies')
class DialogueCompositionScoringTests(unittest.TestCase):
    def setUp(self):
        data = json.loads(DATA.read_text(encoding='utf-8'))
        self.cases = {case['name']: case for case in data['evaluation']}

    def test_updated_hour_requires_the_correct_minutes(self):
        case = self.cases['craft_booking']
        self.assertTrue(parse_answer('周日上午十一点，请带上证件去银杏工坊。', case)['complete'])
        self.assertFalse(parse_answer('周日上午十一点半，请带上证件去银杏工坊。', case)['complete'])
        self.assertFalse(parse_answer('周日上午11:30，请带上证件去银杏工坊。', case)['complete'])
        self.assertFalse(parse_answer('周日晚上十一点，请带上证件去银杏工坊。', case)['complete'])

    def test_quantity_and_requested_format_are_required(self):
        tickets = self.cases['tickets']
        self.assertTrue(parse_answer('灯会票五张，书展票五张。', tickets)['complete'])
        self.assertFalse(parse_answer('灯会票十五张，书展票5张。', tickets)['complete'])
        package = self.cases['package']
        self.assertTrue(parse_answer('金莓 / 两卷 / 花雨巷', package)['complete'])
        self.assertFalse(parse_answer('花雨巷 / 金莓 / 两卷', package)['complete'])

    def test_tiled_supervision_covers_every_answer_position(self):
        for count in range(1, 65):
            tiles = request_tiles([([0], list(range(count)), {})])
            self.assertEqual(sorted(index for _, _, indices in tiles for index in indices), list(range(count)))
            self.assertTrue(all(len(indices) <= 4 for _, _, indices in tiles))
