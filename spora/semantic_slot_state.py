"""Versioned, atomically persisted semantic-slot state."""
import hashlib
import json
from pathlib import Path

import torch

from .language_training import atomic_save


FORMAT = 'spora-semantic-slot-state-v1'


def _digest(metadata, tensors):
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    digest = hashlib.sha256(encoded)
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(',', ':')).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SemanticSlotState:
    """A small immutable-by-publication bank of vectors."""

    def __init__(self, width, max_slots, *, slots=None, occupied=None, revision=0):
        if type(width) is not int or width < 2 or type(max_slots) is not int or not 1 <= max_slots <= 4096:
            raise ValueError('Invalid semantic slot dimensions')
        if type(revision) is not int or revision < 0:
            raise ValueError('Invalid semantic slot revision')
        if slots is None:
            slots = torch.zeros(max_slots, width, dtype=torch.float32)
        if occupied is None:
            occupied = torch.zeros(max_slots, dtype=torch.bool)
        if (not isinstance(slots, torch.Tensor) or slots.shape != (max_slots, width)
                or not slots.is_floating_point() or not isinstance(occupied, torch.Tensor)
                or occupied.shape != (max_slots,) or occupied.dtype != torch.bool
                or not bool(torch.isfinite(slots).all())):
            raise ValueError('Invalid semantic slot tensors')
        self.width = width
        self.max_slots = max_slots
        self.slots = slots.detach().clone()
        self.occupied = occupied.detach().clone()
        self.revision = revision

    def validate(self):
        SemanticSlotState(self.width, self.max_slots, slots=self.slots,
                          occupied=self.occupied, revision=self.revision)

    def put(self, index, vector):
        if type(index) is not int or not 0 <= index < self.max_slots:
            raise ValueError('Semantic slot index is outside the bank')
        if (not isinstance(vector, torch.Tensor) or vector.shape != (self.width,)
                or not vector.is_floating_point() or not bool(torch.isfinite(vector).all())):
            raise ValueError('Semantic slot vector has an invalid shape or value')
        slots = self.slots.clone()
        occupied = self.occupied.clone()
        slots[index] = vector.detach().to(dtype=slots.dtype, device=slots.device)
        occupied[index] = True
        return SemanticSlotState(self.width, self.max_slots, slots=slots,
                                 occupied=occupied, revision=self.revision + 1)

    def get(self, index, *, device=None):
        if type(index) is not int or not 0 <= index < self.max_slots:
            raise ValueError('Semantic slot index is outside the bank')
        if not bool(self.occupied[index]):
            raise KeyError('Semantic slot is empty')
        value = self.slots[index].clone()
        return value if device is None else value.to(device)


def save(path, state, *, base_sha256, bridge_sha256, metadata=None):
    if not isinstance(state, SemanticSlotState) or not isinstance(base_sha256, str) or not base_sha256:
        raise ValueError('State and base binding are required')
    state.validate()
    info = {'format': FORMAT, 'width': state.width, 'max_slots': state.max_slots,
            'revision': state.revision, 'base_sha256': base_sha256,
            'bridge_sha256': bridge_sha256, 'metadata': dict(metadata or {})}
    tensors = {'slots': state.slots, 'occupied': state.occupied}
    atomic_save({'metadata': info, 'tensors': {key: value.cpu().clone() for key, value in tensors.items()},
                 'integrity_sha256': _digest(info, tensors)}, path)
    return _digest(info, tensors)


def load(path, *, expected_base_sha256=None, expected_bridge_sha256=None, device='cpu'):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'tensors', 'integrity_sha256'}:
        raise ValueError('Invalid semantic slot checkpoint')
    metadata, tensors = payload['metadata'], payload['tensors']
    if (not isinstance(metadata, dict) or metadata.get('format') != FORMAT
            or set(tensors) != {'slots', 'occupied'}
            or _digest(metadata, tensors) != payload['integrity_sha256']):
        raise ValueError('Semantic slot checkpoint integrity failure')
    if expected_base_sha256 is not None and metadata.get('base_sha256') != expected_base_sha256:
        raise ValueError('Semantic slot base binding mismatch')
    if expected_bridge_sha256 is not None and metadata.get('bridge_sha256') != expected_bridge_sha256:
        raise ValueError('Semantic slot bridge binding mismatch')
    state = SemanticSlotState(metadata['width'], metadata['max_slots'], slots=tensors['slots'],
                             occupied=tensors['occupied'], revision=metadata['revision'])
    state.slots = state.slots.to(device)
    state.occupied = state.occupied.to(device)
    return state, metadata
