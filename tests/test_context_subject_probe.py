import unittest

from scripts.deepseek_context_subject_probe import compose, current_mask
from scripts.deepseek_local_baseline import MODEL
from transformers import AutoTokenizer


@unittest.skipUnless(MODEL.exists(), 'Local DeepSeek tokenizer is unavailable')
class ContextSubjectProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        cls.tokenizer.padding_side = 'right'

    def test_mask_selects_only_current_turn_in_padded_batch(self):
        turns = [('Bot', '我喜欢喝茶。', 'Usr', '我喜欢咖啡。'),
                 ('Usr', '你好吗？', 'Bot', '你好，我很好。')]
        pairs = [compose(*turn) for turn in turns]
        combined, starts = zip(*pairs)
        currents = [turn[3] for turn in turns]
        masks = current_mask(self.tokenizer, combined, starts, currents)
        formatted = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
                     tokenize=False, add_generation_prompt=True) for text in combined]
        encoded = self.tokenizer(formatted, return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
        for i, current in enumerate(currents):
            spans = encoded['offset_mapping'][i][masks[i]].tolist()
            start = formatted[i].find(combined[i]) + starts[i]
            self.assertEqual(spans[0][0], start)
            self.assertEqual(spans[-1][1], start + len(current))
            self.assertTrue(all(start <= left < right <= start + len(current) for left, right in spans))

    def test_compose_rejects_invalid_turns(self):
        for args in [('Usr', 'a', 'Usr', 'b'), ('other', 'a', 'Usr', 'b'),
                     ('Bot', '', 'Usr', 'b'), ('Bot', 'a', 'Usr', '')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                compose(*args)


if __name__ == '__main__':
    unittest.main()
