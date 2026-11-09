import unittest

import torch

from scripts.deepseek_cluener_span_audit import character_mask, labeled_names
from scripts.deepseek_cluener_locator_train import remove_validation_texts, token_gold


class CluenerAuditTests(unittest.TestCase):
    def test_name_offsets_are_validated_and_deduplicated(self):
        row = {'text': '周明说话', 'entities': [{'label': 'name', 'start_offset': 0, 'end_offset': 2, 'entity': '周明'},
                                                {'label': 'name', 'start_offset': 0, 'end_offset': 2, 'entity': '周明'}]}
        self.assertEqual(labeled_names(row), [(0, 2)])
        row['entities'][0]['entity'] = '错误'
        with self.assertRaises(ValueError):
            labeled_names(row)

    def test_selected_token_offsets_map_to_source_characters(self):
        offsets = torch.tensor([[0, 2], [2, 4], [4, 5], [5, 6]])
        result = character_mask(offsets, torch.tensor([False, True, False, True]), 2, 4)
        self.assertEqual(result.tolist(), [True, True, False, True])

    def test_training_excludes_validation_text_and_detects_crossing_tokens(self):
        train = [{'text': '重叠'}, {'text': '独立'}]
        self.assertEqual(remove_validation_texts(train, [{'text': '重叠'}]), [{'text': '独立'}])
        offsets = torch.tensor([[0, 2], [2, 5], [5, 6]])
        tokens, representable, gold = token_gold(offsets, torch.tensor([False, True, True]), 2, 4, [(0, 2)])
        self.assertEqual(tokens.tolist(), [False, True, False])
        self.assertFalse(representable)
        self.assertEqual(gold.tolist(), [True, True, False, False])


if __name__ == '__main__':
    unittest.main()
