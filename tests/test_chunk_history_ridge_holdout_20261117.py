import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.deepseek_chunk_history_ridge_holdout_20261117 import build_events, fixed_inputs
from spora.structured_text_memory import StructuredTextMemoryCodec


class FixedInputsTests(unittest.TestCase):
    def test_selection_excludes_earlier_cases_and_duplicate_turns(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1] * 24
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.jsonl'
            rows = [{'case_id': f'case-{i:04d}', 'sessions': [{'turns': [
                {'role': 'user', 'content': 'duplicate' if i in (200, 201) else f'turn {i}'}]}]}
                    for i in range(198, 214)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            got = fixed_inputs(path, Tokenizer())
            self.assertEqual(len(got), 11)
            self.assertEqual(got[0]['case_id'], 'case-0200')
            self.assertNotIn('case-0201', [x['case_id'] for x in got])
            self.assertEqual(len({x['text_sha256'] for x in got}), 11)

    def test_fixed_seed_preserves_two_waves_and_event_count(self):
        embedding = torch.randn((64, 64), generator=torch.Generator().manual_seed(42))
        codec = StructuredTextMemoryCodec(embedding, memory_width=90, identity_slots=3, fact_slots=5,
                                          identity_slot_width=15, fact_slot_width=9, eos_token_id=0)
        inputs = [{'tokens': [1 + (n+i) % 60 for n in range(24)]} for i in range(11)]
        identity = [[i+1] for i in range(5)]
        first = build_events(inputs, identity, codec, 1941)
        again = build_events(inputs, identity, codec, 1941)
        self.assertEqual(len(first), 64)
        self.assertEqual([x['key'] for x in first], [x['key'] for x in again])
        self.assertTrue(all(x['version'] == 0 for x in first[:30]))
        self.assertTrue(all(x['version'] == 1 for x in first[30:60]))
        self.assertEqual([x['key'] for x in first[60:]], [0, 1, 33, 34])
