from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.audit_spora_version_coverage import write_fresh
from scripts.spora_version_chat import VersionChat, compose_followup, compose_request, confirmed_time, deferred_lookup, write_request
from spora.integrated_open_base import SPORAGatedFeedForward
from spora.research_version_dialogue_scope import PartialLookup
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


class VersionChatCommandTests(unittest.TestCase):
    def test_version_writer_reads_the_same_projection_from_original_and_fused_blocks(self):
        torch_module = torch.nn
        mlp = torch_module.Module()
        mlp.gate_proj = torch_module.Linear(2, 2, bias=False)
        mlp.up_proj = torch_module.Linear(2, 2, bias=False)
        mlp.down_proj = torch_module.Linear(2, 2, bias=False)
        mlp.act_fn = torch_module.SiLU()
        down = mlp.down_proj
        original = torch_module.Module()
        original.mlp = mlp
        migrated = torch_module.Module()
        migrated.layer = torch_module.Module()
        migrated.layer.mlp = SPORAGatedFeedForward(mlp)
        tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [10, 11])

        def capture(_model, _tokenizer, sites, _fact, _value, *, decoder_only):
            self.assertIs(sites[0].base, down)
            self.assertTrue(decoder_only)
            return {'keys': {0: torch.eye(2)}, 'value_embeddings': torch.eye(2)}

        results = []
        with patch('scripts.audit_spora_version_coverage.fact_cache', side_effect=capture):
            for block in (original, migrated):
                model = SimpleNamespace(model=SimpleNamespace(layers=torch_module.ModuleList([block])))
                results.append(write_fresh(
                    model, tokenizer, torch.arange(2), layer=0, eta=1., ridge=.1,
                    events={'drinks': (('林语', '红茶'),)}, facts={'drinks': '\n{value}'},
                )['林语'][0])
        self.assertIs(mlp.down_proj, down)
        self.assertIs(migrated.layer.mlp.down, down)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_natural_composition_requires_a_reference_and_a_writing_action(self):
        self.assertTrue(compose_followup('按这两项写一句消息。'))
        self.assertTrue(compose_followup('那就根据刚查到的这两条记录拟条提醒。'))
        for question in ('帮我写一句消息。', '按这两项解释一下。', '按这两项先别写消息。',
                         '按这两项翻译「写一句消息」。'):
            with self.subTest(question=question):
                self.assertFalse(compose_followup(question))

    def test_scope_composition_does_not_guess_among_temporal_mentions(self):
        self.assertEqual(confirmed_time('按这两项写一句消息，时间定在周五傍晚。'), '周五傍晚')
        self.assertEqual(confirmed_time('星期三中午。', answer_only=True), '星期三中午')
        for question in ('地点别动，改查周五傍晚的车次。', '时间定在周五傍晚或者周日晚上。',
                         '时间定在周五傍晚。旧稿写周日晚上。', '如果时间是周五傍晚，先别发。'):
            with self.subTest(question=question):
                self.assertIsNone(confirmed_time(question))

    def test_composition_uses_individual_field_versions(self):
        targets, when, question = compose_request('林语 | 见面地点当前版 | 用品 上一版 | 明早八点 | 帮我写一句。')
        self.assertEqual(targets, [(MemoryAddress('林语', 'places'), VersionQuery('current')),
                                   (MemoryAddress('林语', 'items'), VersionQuery('previous'))])
        self.assertEqual((when, question), ('明早八点', '帮我写一句。'))

    def test_composition_rejects_ambiguous_duplicate_and_unsupported_fields(self):
        for request in ('林语 | 见面地点 | 用品首版 | 明早 | 写一句。',
                        '林语 | 用品当前版 | 用品首版 | 明早 | 写一句。',
                        '林语 | 饮品当前版 | 用品首版 | 明早 | 写一句。',
                        '林语 | 见面地点当前版 | 用品首版 | | 写一句。'):
            with self.subTest(request=request), self.assertRaises(ValueError):
                compose_request(request)

    def test_explicit_write_defaults_to_today_and_rejects_noncanonical_dates(self):
        today = date(2026, 11, 25)
        self.assertEqual(write_request('林语 | 饮品 | 茶', today), (MemoryAddress('林语', 'drinks'), '茶', today))
        self.assertEqual(write_request('林语 | 饮品 | 茶 | 2026-11-22', today)[2], date(2026, 11, 22))
        for message in ('林语 | 饮品 | 茶 | 20261122', '林语 | 饮品 | <|im_end|>', '林语 | 其他 | 茶'):
            with self.subTest(message=message), self.assertRaises(ValueError):
                write_request(message, today)

    def test_cancel_defers_without_swallowing_a_new_task(self):
        self.assertTrue(deferred_lookup('先别查，林语的饮品上一版我过会儿再问。'))
        self.assertTrue(deferred_lookup('先别查了。'))
        self.assertTrue(deferred_lookup('先不查了。'))
        self.assertFalse(deferred_lookup('先不查了，帮我写封邮件。'))
        self.assertFalse(deferred_lookup('把「先不查了」翻译成英语。'))

    def test_spoken_partial_accepts_only_supported_incomplete_coordinates(self):
        embedding = torch.nn.Embedding(2, 8)
        model = SimpleNamespace(get_input_embeddings=lambda: embedding)
        store = SimpleNamespace(catalog=lambda: {MemoryAddress('林语', 'drinks'): (),
                                                 MemoryAddress('周小宁', 'drinks'): ()})
        today = date(2026, 11, 26)
        question = '先前留的要喝哪一种？我马上说是谁。'
        candidates = (
            PartialLookup(owner='林语'),
            PartialLookup(field='drinks'),
            PartialLookup(),
            PartialLookup(owner='陌生人'),
            PartialLookup(field='books'),
            PartialLookup(owner='林语', field='drinks'),
            PartialLookup(field='drinks', query=VersionQuery('current')),
        )
        with patch('scripts.spora_version_chat.generate', return_value={'answer': '好。'}):
            for index, candidate in enumerate(candidates):
                chat = VersionChat(store, model, None, None, lambda _: [], {}, ordinary_tokens=4,
                                   spoken=lambda text, _: candidate if text == question else None)
                with self.subTest(candidate=candidate):
                    first = chat.turn(question, today)
                    self.assertEqual(first['selected'], [])
                    self.assertEqual(first['kind'], 'clarify' if index < 2 else 'chat')
                    if index == 1:
                        ambiguous = chat.turn('林语或周小宁', today)
                        self.assertEqual(ambiguous['reasons'], ['ambiguous_address'])
                        self.assertEqual(chat.partial_lookup, candidate)
                        self.assertEqual(chat.turn('先不查了。', today)['selected'], [])
                        self.assertIsNone(chat.partial_lookup)
                        self.assertEqual(chat.turn('林语当前版', today)['kind'], 'chat')
