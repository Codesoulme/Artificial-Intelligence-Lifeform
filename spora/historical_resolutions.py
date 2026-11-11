"""Explicit, bounded adjudications for deferred historical observations.

The records are exact tensor-backed metadata, not learned memory weights or
automatic truth judgments. This module is opt-in and is not a security boundary.
Callers own identity verification, authorization and transactional persistence.
"""
import torch
from torch import nn

from .growing import query
from .internal_memory import read_provenance


class HistoricalResolutions(nn.Module):
    def __init__(self, key_count, capacity=128):
        super().__init__()
        if (type(key_count) is not int or not 1 <= key_count <= 65536
                or type(capacity) is not int or not 1 <= capacity <= 65536):
            raise ValueError('Invalid historical resolution dimensions')
        self.key_count, self.capacity = key_count, capacity
        # key, observed-event, effective-time, value, source (direct/report), decision
        self.register_buffer('records', torch.zeros(capacity, 6, dtype=torch.int64))
        self.register_buffer('occupied', torch.zeros(capacity, dtype=torch.bool))

    def config_dict(self):
        return {'key_count': self.key_count, 'capacity': self.capacity}

    def validate(self, model=None, state=None):
        if (type(self.key_count) is not int or not 1 <= self.key_count <= 65536
                or type(self.capacity) is not int or not 1 <= self.capacity <= 65536
                or self.records.shape != (self.capacity, 6) or self.records.dtype != torch.int64
                or self.occupied.shape != (self.capacity,) or self.occupied.dtype != torch.bool
                or self.records.device != self.occupied.device):
            raise ValueError('Invalid historical resolution tensors')
        if bool((self.records[~self.occupied] != 0).any()):
            raise ValueError('Unoccupied resolution slot contains data')
        if model is not None and model.config.key_count != self.key_count:
            raise ValueError('Resolution keys do not match individual')
        active = self.records[self.occupied]
        if len(active):
            if (bool(((active[:, 0] < 0) | (active[:, 0] >= self.key_count)).any())
                    or bool((active[:, 1] < 0).any()) or bool((active[:, 2] < 0).any())
                    or bool((active[:, 2] > active[:, 1]).any()) or bool((active[:, 3] < 0).any())
                    or bool(((active[:, 4] < 0) | (active[:, 4] > 1)).any())
                    or bool(~((active[:, 5] == 1) | (active[:, 5] == -1)).all())
                    or torch.unique(active[:, :2], dim=0).shape[0] != len(active)):
                raise ValueError('Invalid or duplicate historical resolution')
            if model is not None and bool((active[:, 3] >= model.config.unknown).any()):
                raise ValueError('Resolution value outside finite vocabulary')
            if state is not None and bool((active[:, 1] >= state.cursor).any()):
                raise ValueError('Resolution references an unobserved event')
        return self

    @torch.no_grad()
    def lookup(self, key, event):
        if type(key) is not int or type(event) is not int:
            raise ValueError('Integer key and event required')
        matches = (self.occupied & (self.records[:, 0] == key) & (self.records[:, 1] == event)).nonzero(as_tuple=True)[0]
        if len(matches) > 1:
            raise ValueError('Duplicate historical resolution')
        if not len(matches):
            return None
        row = self.records[int(matches[0])].tolist()
        return {'key': row[0], 'event': row[1], 'valid_at': row[2], 'value': row[3],
                'source': ('direct', 'report')[row[4]], 'decision': 'confirm' if row[5] == 1 else 'reject'}

    @torch.no_grad()
    def verify(self, model, state, key, event, provenance):
        """Recheck the original event; return None on missing or changed evidence."""
        entry = self.lookup(key, event)
        if (entry is None or provenance.get('action') != 'defer' or 'reason' in provenance
                or provenance.get('source') != entry['source'] or provenance.get('valid_at') != entry['valid_at']):
            return None
        device = next(model.parameters()).device
        keys = torch.tensor([key], dtype=torch.long, device=device)
        logits = query(model, state, keys, torch.ones_like(keys), torch.full_like(keys, event))
        if not bool(torch.isfinite(logits).all()) or int(logits.argmax(-1)[0]) != entry['value']:
            return None
        return entry

    @torch.no_grad()
    def record(self, model, state, key, event, *, value, valid_at, decision):
        """Record explicit adjudication after checking the original neural read.

        This mutates the module in place; persist only via a guarded transaction
        on a private copy. No current or history memory tensor is modified.
        """
        self.validate(model, state)
        if (type(key) is not int or not 0 <= key < self.key_count or type(event) is not int
                or not 0 <= event < state.cursor or type(value) is not int or not 0 <= value < model.config.unknown
                or type(valid_at) is not int or not 0 <= valid_at <= event or decision not in ('confirm', 'reject')):
            raise ValueError('Invalid explicit resolution')
        if self.lookup(key, event) is not None:
            raise ValueError('Event already resolved')
        provenance = read_provenance(model, state, key, event)
        if (provenance.get('action') != 'defer' or provenance.get('source') not in ('direct', 'report')
                or provenance.get('valid_at') != valid_at or 'reason' in provenance):
            raise ValueError('Deferred factual provenance does not match')
        device = next(model.parameters()).device
        keys = torch.tensor([key], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = query(model, state, keys, torch.ones_like(keys), torch.full_like(keys, event))
        if not bool(torch.isfinite(logits).all()) or int(logits.argmax(-1)[0]) != value:
            raise ValueError('Historical value cannot be verified')
        free = (~self.occupied).nonzero(as_tuple=True)[0]
        if not len(free):
            raise ValueError('Historical resolution capacity exhausted')
        slot = int(free[0])
        source = 0 if provenance['source'] == 'direct' else 1
        self.records[slot] = torch.tensor([key, event, valid_at, value, source,
                                           1 if decision == 'confirm' else -1],
                                          dtype=torch.int64, device=self.records.device)
        self.occupied[slot] = True
