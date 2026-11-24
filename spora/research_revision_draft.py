"""Buffer and apply revisions according to the latest relevant decision."""

from dataclasses import dataclass
from enum import IntEnum


class RevisionDecision(IntEnum):
    PROPOSE = 0
    CONFIRM = 1
    WITHDRAW = 2
    OTHER = 3


@dataclass(frozen=True)
class RevisionDraft:
    """One scoped revision buffered until the current transaction is applied."""

    field: str
    value: str | None = None
    ready: bool = False

    def observe(self, action, *, field=None, value=None):
        if value is not None and field is None:
            raise ValueError('An explicit value must name its field')
        if action == RevisionDecision.OTHER or (field is not None and field != self.field):
            return self
        if action == RevisionDecision.WITHDRAW:
            return RevisionDraft(self.field)
        if action not in (RevisionDecision.PROPOSE, RevisionDecision.CONFIRM):
            raise ValueError('Unknown revision decision')
        current = self.value if value is None else value
        return RevisionDraft(self.field, current, bool(current) and action == RevisionDecision.CONFIRM)

    def patch(self):
        """Return the write for the separately initiated apply operation."""
        return {self.field: self.value} if self.ready and self.value is not None else {}
