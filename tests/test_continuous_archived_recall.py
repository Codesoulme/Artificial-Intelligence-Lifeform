import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import deepseek_continuous_memory_chat as runtime


class ArchivedRecallTests(unittest.TestCase):
    def answer(self, generated, *, back=1, as_of=None, archive_present=True):
        chat = runtime.ContinuousMemoryChat.__new__(runtime.ContinuousMemoryChat)
        chat.addresses = {'subject': 1}
        chat.working_path = Path('memory.sqlite3')
        chat.working_sha = 'revision'
        chat.position_limit = 16
        chat.self_name = chat.ordinary_intent = None
        chat.inheritance = chat.direct = chat.decoder = None
        chat.tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda *_args, **_kwargs: 'archived value')
        chat.styles = {'drink': None}
        chat.columns = chat.timing = chat.codes = None
        chat.scale = 1.
        chat.cutoff = chat.gate_cutoff = .5
        chat.filled = {(1, 'drink', 1)}
        archive = torch.zeros(runtime.SLOTS, runtime.RANK)
        chat.bank_specs = {('drink', 1): (archive.shape, archive.dtype)}
        chat.site = None
        chat.bank_for = lambda *_args: None
        chat._local_address = lambda _: 1
        bias = torch.tensor([[0., 3., 0.], [3., 0., 0.]])
        with (patch.object(runtime, 'features_for', return_value=(None, None, None, [1, 2])),
              patch.object(runtime, 'gate_probabilities', return_value=[0.]),
              patch.object(runtime, 'route', return_value=[dict(selected='drink', confidence=.99)]),
              patch.object(runtime, 'route_action', return_value='read'),
              patch.object(runtime, 'read_archived_slots', return_value=archive if archive_present else None),
              patch.object(runtime, 'read_slots_as_of', return_value=archive if archive_present else None),
              patch.object(runtime, 'cosine_slot_bias', return_value=bias),
              patch.object(runtime, 'cosine_bias', return_value=bias),
              patch.object(runtime, 'generate_with_memory', return_value=([2, 0], None)),
              patch.object(runtime, 'generate_memory_suffix', return_value=generated) as direct):
            result = chat.answer('question', entity=1, back=back, as_of=as_of)
        return result, direct.call_count

    def test_verified_archive_is_read_when_timing_gate_misses(self):
        result, calls = self.answer([1, 0])
        self.assertEqual((result['action'], result['response'], result['memory_start']), ('read', 'archived value', 0))
        self.assertEqual(calls, 1)

    def test_archive_requires_matching_tokens_and_eos(self):
        for generated in ([2, 0], [1]):
            with self.subTest(generated=generated):
                result, calls = self.answer(generated)
                self.assertEqual(result['action'], 'clarify')
                self.assertEqual(calls, 1)

    def test_current_read_preserves_timing_gate(self):
        result, calls = self.answer([1, 0], back=0)
        self.assertEqual((result['action'], result['reason']), ('clarify', 'injection_not_triggered'))
        self.assertEqual(calls, 0)

    def test_as_of_read_uses_verified_archive(self):
        result, calls = self.answer([1, 0], back=0, as_of=0)
        self.assertEqual(result['action'], 'read')
        self.assertEqual(result['as_of_utc'], '1970-01-01T00:00:00+00:00')
        self.assertIn('archived value', result['response'])
        self.assertEqual(calls, 1)

    def test_absent_archive_does_not_generate_a_value(self):
        result, calls = self.answer([1, 0], archive_present=False)
        self.assertEqual((result['action'], result['reason']), ('missing', 'history_unavailable'))
        self.assertEqual(calls, 0)
