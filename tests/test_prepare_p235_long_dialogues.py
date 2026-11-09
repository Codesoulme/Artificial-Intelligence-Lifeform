import unittest

from scripts.prepare_p235_long_dialogues import split_dialogue


class Tokenizer:
    def token_to_id(self, text):
        return {'<|bos|>': 1, '<|eos|>': 2, '<|user|>': 3, '<|assistant|>': 4}[text]

    def encode(self, text):
        return type('Encoding', (), {'ids': [ord(c) for c in text]})()


class LongDialogueTests(unittest.TestCase):
    def test_history_is_trimmed_by_turn_and_targets_not_repeated(self):
        messages = [{'role': role, 'content': value} for role, value in
                    [('user', 'aa'), ('assistant', 'bb'), ('user', 'cc'), ('assistant', 'dd')]]
        rows, skipped = split_dialogue(Tokenizer(), messages, limit=9)
        self.assertEqual(skipped, [])
        self.assertEqual([r['context_start_message_index'] for r in rows], [0, 2])
        self.assertEqual([[t for t, enabled in zip(r['tokens'], r['mask']) if enabled] for r in rows],
                         [[ord('b'), ord('b'), 2], [ord('d'), ord('d'), 2]])

    def test_complete_answer_too_long_is_skipped(self):
        messages = [{'role': 'user', 'content': 'aa'}, {'role': 'assistant', 'content': 'bbbbbb'}]
        self.assertEqual(split_dialogue(Tokenizer(), messages, limit=9), ([], [1]))

    def test_earlier_answers_are_context_without_duplicate_supervision(self):
        messages = [{'role': role, 'content': value} for role, value in
                    [('user', 'a'), ('assistant', 'b'), ('user', 'c'), ('assistant', 'd')]]
        rows, skipped = split_dialogue(Tokenizer(), messages, limit=20)
        self.assertEqual(skipped, [])
        self.assertEqual(rows[1]['context_start_message_index'], 0)
        self.assertEqual([t for t, flag in zip(rows[1]['tokens'], rows[1]['mask']) if flag], [ord('d'), 2])

    def test_rejects_bad_roles(self):
        with self.assertRaises(ValueError):
            split_dialogue(Tokenizer(), [{'role': 'assistant', 'content': 'a'}, {'role': 'user', 'content': 'b'}])


if __name__ == '__main__':
    unittest.main()
