from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from scripts.spora_named_memory_chat import LocalChat, PendingFact
from spora.integrated_durable import StructuredRecordChanged


class TokenBatch(dict):
    def to(self, _device):
        return self


class NamedChatPositionTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(position_table_max_position=110, generate_ordinary=Mock())
        self.tokenizer = Mock(pad_token_id=0, eos_token_id=1)
        self.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: ''.join(
            message['content'] for message in messages
        )
        self.tokenizer.side_effect = lambda prompt, **kwargs: TokenBatch(
            input_ids=torch.ones(1, len(prompt), dtype=torch.long)
        )
        self.tokenizer.decode.return_value = '收到'
        self.runtime.generate_ordinary.return_value = torch.ones(1, 18, dtype=torch.long)
        self.chat = LocalChat(self.runtime, self.tokenizer, None, None)

    def test_trims_old_exchange_and_leaves_room_for_reply(self):
        self.chat.turns = [{'role': 'user', 'content': '很长的上一条消息'}, {'role': 'assistant', 'content': '收到'}]
        self.assertEqual(self.chat.talk('你好'), '收到')
        kwargs = self.runtime.generate_ordinary.call_args.kwargs
        self.assertLessEqual(kwargs['input_ids'].shape[1] + kwargs['max_new_tokens'], 110)
        self.assertEqual([turn['content'] for turn in self.chat.turns], ['你好', '收到'])

    def test_rejects_input_that_cannot_leave_reply_budget(self):
        self.assertIn('太长', self.chat.talk('一个无法放进位置表的提问'))
        self.runtime.generate_ordinary.assert_not_called()

    def test_stale_pending_fact_is_cancelled_with_a_localized_message(self):
        self.runtime.write_structured_relation = Mock(side_effect=StructuredRecordChanged())
        self.chat.pending_fact = PendingFact(4, '集合点', torch.ones(64), {}, {}, 3)
        reply = self.chat.handle('/确认记事')
        self.assertIn('请重新提出修改', reply)
        self.assertIsNone(self.chat.pending_fact)
        self.assertEqual(self.runtime.write_structured_relation.call_args.kwargs['expected_record_revision'], 3)
        self.assertEqual(self.chat.handle('/确认记事'), '没有待确认的事实。')
