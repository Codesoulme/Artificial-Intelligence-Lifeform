"""Session-local proposal handles over the existing revision-checked write path."""

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from spora.integrated_durable import StructuredRecordChanged


@dataclass(frozen=True)
class Proposal:
    session: str
    number: int
    scope: tuple[int, str]
    issued: int
    valid_through: int


class PendingOffers:
    def __init__(self, runtime):
        self.runtime = runtime
        self.session = uuid4().hex
        self.turn = 0
        self.next_number = 1
        self.active = {}

    def advance(self, turn):
        if type(turn) is not int or turn < self.turn:
            raise ValueError('Proposal turns must be nondecreasing integers')
        self.turn = turn
        self.active = {scope: entry for scope, entry in self.active.items() if entry[0].valid_through >= turn}

    def propose(self, prepared, *, turn, valid_through):
        if type(turn) is not int or type(valid_through) is not int or valid_through < turn:
            raise ValueError('A proposal needs a valid closing turn')
        self.advance(turn)
        scope = (prepared.entity, prepared.relation)
        handle = Proposal(self.session, self.next_number, scope, turn, valid_through)
        self.next_number += 1
        self.active[scope] = handle, deepcopy(prepared)
        return handle

    def _lookup(self, handle, scope, turn):
        self.advance(turn)
        if handle is None:
            return 'unbound', None
        if handle.session != self.session:
            return 'other_session', None
        if handle.scope != scope:
            return 'wrong_scope', None
        if turn < handle.issued or turn > handle.valid_through:
            return 'expired', None
        current = self.active.get(scope)
        if current is None:
            return 'closed', None
        if current[0] != handle:
            return 'superseded', None
        return 'open', current[1]

    def withdraw(self, handle, *, scope, turn):
        status, pending = self._lookup(handle, scope, turn)
        if pending is None:
            return status
        del self.active[scope]
        return 'withdrawn'

    def confirm(self, handle, *, scope, turn, authorized):
        status, pending = self._lookup(handle, scope, turn)
        if pending is None:
            return status, None
        if authorized is not True:
            return 'held', None
        try:
            result = self.runtime.write_structured_relation(
                pending.entity, pending.relation, pending.packet, pending.bounds,
                codec_config=pending.codec_config, replace_existing=True, retain_previous=True,
                expected_record_revision=pending.revision)
        except StructuredRecordChanged:
            del self.active[scope]
            return 'revision_changed', None
        del self.active[scope]
        return 'committed', result['record_revision']
