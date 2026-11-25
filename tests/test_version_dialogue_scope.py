import unittest

from spora.research_version_dialogue_scope import (advance_scope, dialogue_targets, pending_after_lookup,
                                                   pending_followup_targets)
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


FIELDS = {'drink': '饮品', 'place': '见面地点'}
NAMES = ('闻韶宁', '闻绍宁')
DRINK = MemoryAddress('闻韶宁', 'drink')
PLACE = MemoryAddress('闻韶宁', 'place')
OTHER_DRINK = MemoryAddress('闻绍宁', 'drink')


class DialogueVersionScopeTests(unittest.TestCase):
    def test_bare_followup_uses_confirmed_address_but_selects_its_own_version(self):
        target, = dialogue_targets('那上一版呢？', NAMES, FIELDS, (DRINK,))
        self.assertEqual(target.address, DRINK)
        self.assertEqual(target.query, VersionQuery('previous'))
        self.assertEqual(target.source, 'carried')
        self.assertEqual(advance_scope((PLACE,), (target,), [7]), (DRINK,))

    def test_named_field_can_shift_to_another_field_of_the_same_person(self):
        target, = dialogue_targets('那见面地点当前版呢？', NAMES, FIELDS, (DRINK,))
        self.assertEqual(target.address, PLACE)
        self.assertEqual(target.query, VersionQuery('current'))

    def test_pair_needs_a_quantifier_or_an_unambiguous_field(self):
        ordered = (DRINK, PLACE)
        target, = dialogue_targets('那上一版呢？', NAMES, FIELDS, ordered)
        self.assertIsNone(target.address)
        targets = dialogue_targets('两项都查首版。', NAMES, FIELDS, ordered)
        self.assertEqual([item.address for item in targets], list(ordered))
        ambiguous, = dialogue_targets('那饮品上一版呢？', NAMES, FIELDS, (DRINK, OTHER_DRINK))
        self.assertIsNone(ambiguous.address)

    def test_unknown_explicit_owner_does_not_fall_back_to_the_previous_owner(self):
        for question in ('那闻韶蔚的饮品上一版呢？', '那闻绍宁的当前版呢？', '那他的是哪一版？'):
            with self.subTest(question=question):
                target, = dialogue_targets(question, NAMES, FIELDS, (DRINK,))
                self.assertIsNone(target.address)
                self.assertEqual(advance_scope((DRINK,), (target,), []), ())

    def test_nonread_and_partial_lookup_cannot_establish_a_new_scope(self):
        quoted = '把「那上一版呢」翻译成英语。'
        self.assertEqual(dialogue_targets(quoted, NAMES, FIELDS, (DRINK,)), ())
        self.assertEqual(advance_scope((DRINK,), (), []), ())
        self.assertEqual(advance_scope((DRINK,), (), [], persistent=True), (DRINK,))
        targets = dialogue_targets('两项都查上一版。', NAMES, FIELDS, (DRINK, OTHER_DRINK))
        self.assertEqual(advance_scope((DRINK, OTHER_DRINK), targets, []), ())

    def test_standalone_cancellation_does_not_read_a_subsequent_reference(self):
        utterance = '先别查，闻韶宁的饮品上一版我过会儿再问。'
        self.assertEqual(dialogue_targets(utterance, NAMES, FIELDS, (DRINK,)), ())

    def test_explicit_correction_replaces_inherited_owner(self):
        text = '不是闻韶宁的饮品，改查闻绍宁的饮品当前版。'
        target, = dialogue_targets(text, NAMES, FIELDS, (DRINK,))
        self.assertEqual(target.address, OTHER_DRINK)
        self.assertEqual(target.source, 'named')

    def test_pending_retains_only_one_known_unresolved_address(self):
        catalog = {DRINK: (), PLACE: ()}
        question = '查闻韶宁的饮品首版还是现行版？'
        targets = dialogue_targets(question, NAMES, FIELDS, ())
        self.assertEqual(pending_after_lookup(catalog, targets, [], ('unresolved_scope',)), DRINK)
        self.assertIsNone(pending_after_lookup(catalog, targets, [], ('missing_version',)))
        self.assertIsNone(pending_after_lookup(catalog, (*targets, *targets), [],
                                              ('unresolved_scope', 'unresolved_scope')))

    def test_pending_accepts_a_version_and_cannot_transfer_or_renew(self):
        target, = pending_followup_targets('那首版呢？', NAMES, FIELDS, DRINK)
        self.assertEqual(target.address, DRINK)
        self.assertEqual(target.source, 'pending')
        self.assertEqual(target.query, VersionQuery('ordinal', 1))
        changed, = pending_followup_targets('那见面地点首版呢？', NAMES, FIELDS, DRINK)
        self.assertIsNone(changed.address)
        unresolved = pending_followup_targets('那之前旧版本是哪份呢？', NAMES, FIELDS, DRINK)
        self.assertIsNone(pending_after_lookup({DRINK: ()}, unresolved, [], ('unresolved_scope',)))

    def test_pending_accepts_the_short_command_offered_by_the_chat_prompt(self):
        for text, expected in (('查当前版。', VersionQuery('current')), ('首版', VersionQuery('ordinal', 1)),
                               ('就查上一版吧', VersionQuery('previous'))):
            with self.subTest(text=text):
                target, = pending_followup_targets(text, NAMES, FIELDS, DRINK)
                self.assertEqual(target.address, DRINK)
                self.assertEqual(target.query, expected)
