"""Regression coverage for proposal handle validity and conditional writes."""

from types import SimpleNamespace
import unittest

from spora.integrated_durable import StructuredRecordChanged
from spora.research_pending_offers import PendingOffers


class Store:
    def __init__(self):
        self.revisions = {}
        self.writes = []
        self.fail = False

    def write_structured_relation(self, entity, relation, packet, bounds, *, expected_record_revision, **kwargs):
        if self.fail:
            raise OSError('temporary failure')
        key = (entity, relation)
        if self.revisions.get(key) != expected_record_revision:
            raise StructuredRecordChanged('new record')
        revision = len(self.writes) + 1
        self.revisions[key] = revision
        self.writes.append((key, packet, revision))
        return {'record_revision': revision}


def draft(entity, relation, packet=None):
    return SimpleNamespace(entity=entity, relation=relation, packet=packet, bounds={}, codec_config={}, revision=None)


class PendingOfferTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.book = PendingOffers(self.store)

    def test_interleaving_and_replacement_require_correct_handle(self):
        old = self.book.propose(draft(3, '集合点'), turn=1, valid_through=8)
        other = self.book.propose(draft(4, '零食'), turn=2, valid_through=8)
        latest = self.book.propose(draft(3, '集合点'), turn=3, valid_through=8)
        self.assertEqual(self.book.confirm(old, scope=old.scope, turn=4, authorized=True), ('superseded', None))
        self.assertEqual(self.book.confirm(other, scope=old.scope, turn=4, authorized=True), ('wrong_scope', None))
        self.assertEqual(self.book.confirm(None, scope=old.scope, turn=4, authorized=True), ('unbound', None))
        self.assertEqual(self.book.confirm(other, scope=other.scope, turn=5, authorized=True), ('committed', 1))
        self.assertEqual(self.book.confirm(latest, scope=latest.scope, turn=6, authorized=True), ('committed', 2))
        self.assertEqual(self.book.confirm(other, scope=other.scope, turn=7, authorized=True), ('closed', None))
        self.assertEqual([row[0] for row in self.store.writes], [other.scope, latest.scope])

    def test_expiry_withdrawal_and_new_session(self):
        expired = self.book.propose(draft(2, '集合点'), turn=1, valid_through=2)
        self.assertEqual(self.book.confirm(expired, scope=expired.scope, turn=3, authorized=True), ('expired', None))
        withdrawn = self.book.propose(draft(2, '集合点'), turn=4, valid_through=6)
        self.assertEqual(self.book.withdraw(withdrawn, scope=withdrawn.scope, turn=5), 'withdrawn')
        self.assertEqual(self.book.confirm(withdrawn, scope=withdrawn.scope, turn=5, authorized=True), ('closed', None))
        open_offer = self.book.propose(draft(2, '集合点'), turn=6, valid_through=8)
        other = PendingOffers(self.store)
        self.assertEqual(other.confirm(open_offer, scope=open_offer.scope, turn=7, authorized=True), ('other_session', None))
        self.assertEqual(self.store.writes, [])

    def test_revision_conflict_and_transient_retry(self):
        packet = [4, 7]
        first = self.book.propose(draft(1, '集合点', packet), turn=1, valid_through=4)
        packet.append(9)
        self.assertEqual(self.book.confirm(first, scope=first.scope, turn=2, authorized=False), ('held', None))
        self.store.fail = True
        with self.assertRaises(OSError):
            self.book.confirm(first, scope=first.scope, turn=3, authorized=True)
        self.store.fail = False
        self.assertEqual(self.book.confirm(first, scope=first.scope, turn=4, authorized=True), ('committed', 1))
        self.assertEqual(self.store.writes[-1][1], [4, 7])
        conflict = self.book.propose(draft(1, '零食'), turn=5, valid_through=8)
        self.store.revisions[conflict.scope] = 5
        self.assertEqual(self.book.confirm(conflict, scope=conflict.scope, turn=6, authorized=True), ('revision_changed', None))
        self.assertEqual(self.book.confirm(conflict, scope=conflict.scope, turn=7, authorized=True), ('closed', None))
        self.assertEqual(len(self.store.writes), 1)


if __name__ == '__main__':
    unittest.main()
