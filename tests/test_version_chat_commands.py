from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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

    def test_standalone_lookup_cancellation_keeps_questions_and_continuation_distinct(self):
        for question in ('本次调档无须接着办。', '这轮查阅到此为止。', '本轮调阅就此结束！',
                         '这次查询不用再查。'):
            with self.subTest(question=question):
                self.assertTrue(deferred_lookup(question))
        for question in ('本次调档无须接着办吗？', '这轮查阅不要停止。', '本轮调阅不能就此结束。',
                         '他说本轮调阅就此结束。', '把「这轮查阅到此为止」翻译成英文。',
                         '本次调档无须接着办，帮我整理书桌。'):
            with self.subTest(question=question):
                self.assertFalse(deferred_lookup(question))

    def test_archive_opt_out_does_not_carry_addresses_or_prior_record_text(self):
        address = MemoryAddress('林语', 'drinks')
        embedding = torch.nn.Embedding(2, 8)
        store = Mock()
        store.catalog.return_value = {address: ()}
        spoken = Mock(return_value=address)
        chat = VersionChat(store, SimpleNamespace(get_input_embeddings=lambda: embedding),
                           None, None, lambda _: [], {}, ordinary_tokens=4, spoken=spoken)
        chat.history = [{'role': 'user', 'content': '请查林语的饮品'}, {'role': 'assistant', 'content': '茉莉茶'}]
        chat.scope, chat.pending, chat.spoken_pending = (address,), address, address
        question = '不需要翻档。讨论一下怎样给饮品分类。'
        with patch('scripts.spora_version_chat.generate', return_value={'answer': '可以按原料分类。'}) as generate:
            result = chat.turn(question, date(2026, 11, 26))
        self.assertEqual((result['kind'], result['selected']), ('chat', []))
        self.assertEqual(generate.call_args.kwargs['history'], [])
        self.assertIsNone(chat.pending)
        self.assertIsNone(chat.spoken_pending)
        self.assertEqual(chat.scope, ())
        store.read.assert_not_called()
        spoken.assert_not_called()

    def test_pair_waits_for_both_versions_and_retains_the_first_choice(self):
        today = date(2026, 11, 26)
        drinks, books = MemoryAddress('林语', 'drinks'), MemoryAddress('周小宁', 'books')
        first, current = (SimpleNamespace(number=number, effective=today, observed=today) for number in (1, 2))
        old_book, new_book = (SimpleNamespace(number=number, effective=today, observed=today) for number in (3, 4))
        store = SimpleNamespace(catalog=lambda: {drinks: (first, current), books: (old_book, new_book)})
        embedding = torch.nn.Embedding(2, 8)
        model = SimpleNamespace(get_input_embeddings=lambda: embedding)
        chat = VersionChat(store, model, SimpleNamespace(eos_token_id=9), None, lambda _: [], {}, ordinary_tokens=4)
        with patch('scripts.spora_version_chat.generate', return_value={'answer': '好。'}), patch(
                'scripts.spora_version_chat.reply_to_route', return_value=([1, 4], {'answer': '两项。', 'token_ids': []})) as read:
            for question, followup in (('查林语的饮品；查周小宁的书目。', '林语的饮品'),
                                       ('首版还是当前版？', '林语的饮品'),
                                       ('第9版。', '林语的饮品'),
                                       ('首版。', '周小宁的书目')):
                result = chat.turn(question, today)
                self.assertEqual((result['kind'], result['selected']), ('clarify', []))
                self.assertIn(followup, result['answer'])
            read.assert_not_called()
            self.assertEqual(chat.turn('当前版。', today)['selected'], [1, 4])
            read.assert_called_once()
            self.assertEqual(read.call_args.args[6], [1, 4])

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

    def test_spoken_switch_drops_old_address_and_confirms_model_proposal(self):
        today = date(2026, 11, 26)
        old, new = MemoryAddress('林语', 'drinks'), MemoryAddress('周小宁', 'places')
        catalog = {address: (SimpleNamespace(number=i, effective=today, observed=today),)
                   for i, address in enumerate((old, new), 1)}
        model = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(2, 8))
        planner = Mock(return_value=new)
        chat = VersionChat(SimpleNamespace(catalog=lambda: catalog), model, SimpleNamespace(eos_token_id=9),
                           None, lambda _: [], {}, ordinary_tokens=4, spoken=planner)
        chat.scope = (old,)
        with patch('scripts.spora_version_chat.reply_to_route',
                   return_value=([2], {'answer': '见面地点：云桥。', 'token_ids': []})) as read:
            first = chat.turn('改查周小宁约见时定的地方。', today)
            self.assertEqual(first['selected'], [])
            self.assertIn('周小宁的见面地点', first['answer'])
            self.assertEqual(chat.scope, ())
            self.assertEqual(chat.turn('首版。', today)['selected'], [])
            read.assert_not_called()
            self.assertEqual(chat.turn('是，首版。', today)['selected'], [2])
            read.assert_called_once()
            self.assertEqual(read.call_args.args[6], [2])

    def test_spoken_switch_asks_when_target_is_uncertain_without_inheriting_old_address(self):
        today = date(2026, 11, 26)
        old = MemoryAddress('林语', 'drinks')
        planner = Mock(return_value=None)
        model = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(2, 8))
        chat = VersionChat(SimpleNamespace(catalog=lambda: {old: ()}), model, None, None,
                           lambda _: [], {}, ordinary_tokens=4, spoken=planner)
        chat.scope = (old,)
        reply = chat.turn('换查周小宁那本书。', today)
        self.assertEqual((reply['kind'], reply['selected']), ('clarify', []))
        self.assertEqual(chat.scope, ())
        self.assertEqual(chat.partial_lookup, PartialLookup())
        planner.assert_called_once_with('周小宁那本书。', {'林语'})
