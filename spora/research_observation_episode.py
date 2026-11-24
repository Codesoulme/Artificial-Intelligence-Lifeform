"""Serializable working-state episodes whose read keys contain no conversation text."""

from dataclasses import dataclass

import torch

from .research_dialogue_working_state import FIELDS, WorkingTurnState


@dataclass(frozen=True)
class ObservationEpisode:
    state: WorkingTurnState
    topic: str
    key: torch.Tensor
    revision: int = 0

    def apply(self, changes):
        new = self.state.update(changes)
        return ObservationEpisode(new, self.topic, self.key, self.revision + int(new != self.state))

    def payload(self):
        return {'slots': [list(slot) for slot in self.state.slots], 'topic': self.topic,
                'key': self.key.detach().cpu().clone(), 'revision': self.revision}

    @classmethod
    def load(cls, payload, *, topics, width, vocab):
        slots, topic, key, revision = (payload[name] for name in ('slots', 'topic', 'key', 'revision'))
        if (topic not in topics or len(slots) != len(FIELDS) or type(revision) is not int or revision < 0
                or not isinstance(key, torch.Tensor) or key.shape != (width,) or not key.is_floating_point()
                or not torch.isfinite(key).all()):
            raise ValueError('Malformed stored episode')
        for slot in slots:
            if any(type(token) is not int or not 0 <= token < vocab for token in slot):
                raise ValueError('Stored episode has an invalid token')
        state = WorkingTurnState().update({field: tuple(slot) for field, slot in zip(FIELDS, slots, strict=True) if slot})
        return cls(state, topic, key.detach().cpu().clone(), revision)
