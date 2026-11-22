from collections import deque
import unittest
from unittest.mock import patch

import torch

try:
    import transformers
except ModuleNotFoundError:
    transformers = None

if transformers is not None:
    from scripts.spora_calendar_history_chat import _chat_inputs, session_turn


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return ''.join(message['content'] for message in messages) + '<think>\n'

    def __call__(self, text, **kwargs):
        return {'input_ids': torch.zeros(1, len(text), dtype=torch.long)}


@unittest.skipIf(transformers is None, 'requires optional open-base dependencies')
class MixedCalendarChatTests(unittest.TestCase):
    def test_context_removes_complete_turns_and_keeps_current_question(self):
        tokenizer = FakeTokenizer()
        history = deque([('第一条问题', '第一条回复'), ('第二条', '回复二')])
        inputs, omitted = _chat_inputs(tokenizer, history, '当前问题', 32)
        self.assertIsNotNone(inputs)
        self.assertEqual(omitted, 1)
        self.assertEqual([message['content'] for message in tokenizer.messages], ['第二条', '回复二', '当前问题'])
        self.assertEqual(_chat_inputs(tokenizer, history, '当前问题', 5), (None, 2))

    def test_memory_turns_stay_out_of_the_ordinary_chat_history(self):
        history = deque([('周五有安排', '我知道了')])
        memory = {'routing': {'reason': 'routed'}, 'selection': {}, 'generated': {'answer': '咖啡'}}
        unsupported = {'routing': {'reason': 'unsupported_date_question'}, 'selection': None, 'generated': None}
        reset = {'routing': {'reason': 'date_focus_reset'}, 'selection': None, 'generated': None}
        with patch('scripts.spora_calendar_history_chat.calendar_turn', side_effect=[
            (memory, 2), (unsupported, None), (unsupported, None), (reset, None),
        ]), patch('scripts.spora_calendar_history_chat.ordinary_reply', return_value={'answer': '继续聊安排'}) as reply:
            args = (None, None, None, None, None)
            self.assertEqual(session_turn(*args, '查饮品', None, history), (memory, 2))
            self.assertEqual(session_turn(*args, '小林在2026-01-02和2026-01-03的饮品分别是什么？', 2, history),
                             (unsupported, None))
            reply.assert_not_called()
            self.assertEqual(list(history), [('周五有安排', '我知道了')])
            result, _ = session_turn(*args, '回到周五安排', None, history)
            self.assertEqual(result['routing']['reason'], 'ordinary_chat')
            self.assertEqual(list(history)[-1], ('回到周五安排', '继续聊安排'))
            session_turn(*args, '/reset', None, history)
            self.assertFalse(history)
