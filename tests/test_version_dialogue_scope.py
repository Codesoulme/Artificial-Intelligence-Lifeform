import unittest

from spora.research_version_dialogue_scope import (PartialLookup, advance_scope, ambiguous_partial_reply, continue_partial_lookup,
                                                   dialogue_targets, pending_after_lookup, pending_followup_targets,
                                                   start_partial_lookup)
from spora.research_version_store import MemoryAddress
from spora.research_version_timeline import VersionQuery


FIELDS = {'drink': '饮品', 'place': '见面地点'}
NAMES = ('闻韶宁', '闻绍宁')
DRINK = MemoryAddress('闻韶宁', 'drink')
PLACE = MemoryAddress('闻韶宁', 'place')
OTHER_DRINK = MemoryAddress('闻绍宁', 'drink')


class DialogueVersionScopeTests(unittest.TestCase):
    def test_explicit_partial_lookup_retains_field_and_version_until_owner_is_supplied(self):
        previous = VersionQuery('previous')
        partial = start_partial_lookup('请查谁的饮品上一版。', NAMES, FIELDS)
        self.assertEqual(partial, PartialLookup(field='drink', query=previous))
        filled = continue_partial_lookup('是闻绍宁。', partial, NAMES, FIELDS)
        self.assertEqual(filled.target().address, OTHER_DRINK)
        self.assertEqual(filled.target().query, previous)

    def test_partial_lookup_pairs_a_correction_with_the_previously_named_owner(self):
        partial = start_partial_lookup('查一下闻韶宁的记录。', NAMES, FIELDS)
        corrected = continue_partial_lookup('不是闻韶宁，是闻绍宁。', partial, NAMES, FIELDS)
        self.assertEqual(corrected, PartialLookup(owner='闻绍宁'))
        filled = continue_partial_lookup('饮品当前版。', corrected, NAMES, FIELDS)
        self.assertEqual(filled.target().address, OTHER_DRINK)
        self.assertIsNone(continue_partial_lookup('不是闻绍宁，是闻韶宁。', partial, NAMES, FIELDS))

    def test_partial_lookup_does_not_treat_other_requests_as_slot_answers(self):
        partial = start_partial_lookup('查饮品首版。', NAMES, FIELDS)
        for question in ('闻韶宁还是闻绍宁？', '翻译“闻绍宁”。', '先别查闻绍宁。', '闻绍安。', '帮我推荐饮品。'):
            with self.subTest(question=question):
                self.assertIsNone(continue_partial_lookup(question, partial, NAMES, FIELDS))
        for question in ('听说过闻韶宁的记录吗？', '查一下天气。', '请查闻韶宁的饮品首版。'):
            self.assertIsNone(start_partial_lookup(question, NAMES, FIELDS))

    def test_partial_reply_can_ask_the_user_to_choose_one_coordinate(self):
        self.assertEqual(ambiguous_partial_reply('闻韶宁还是闻绍宁？', NAMES, FIELDS), 'owner')
        self.assertEqual(ambiguous_partial_reply('见面地点或饮品。', NAMES, FIELDS), 'field')
        self.assertEqual(ambiguous_partial_reply('先查首版或者当前版？', NAMES, FIELDS), 'query')
        self.assertEqual(ambiguous_partial_reply('首版或者当前版？', NAMES, FIELDS), 'query')

    def test_full_new_address_without_version_discards_the_old_version(self):
        partial = start_partial_lookup('查闻韶宁的首版。', NAMES, FIELDS)
        changed = continue_partial_lookup('查闻绍宁的饮品。', partial, NAMES, FIELDS)
        self.assertEqual(changed, PartialLookup(owner='闻绍宁', field='drink'))

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
