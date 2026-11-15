"""Atomically persisted sequences of semantic vectors."""
import hashlib
import json
from pathlib import Path

import torch

from .language_training import atomic_save


FORMAT = 'spora-semantic-phrase-slot-state-v1'


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


class SemanticPhraseSlotState:
    """A bounded bank where each slot stores a short vector sequence."""

    def __init__(self, width, max_slots, max_tokens=16, *, slots=None, lengths=None,
                 occupied=None, revision=0):
        if (type(width) is not int or width < 2 or type(max_slots) is not int
                or not 1 <= max_slots <= 4096 or type(max_tokens) is not int
                or not 1 <= max_tokens <= 128):
            raise ValueError('Invalid semantic phrase slot dimensions')
        if type(revision) is not int or revision < 0:
            raise ValueError('Invalid semantic phrase slot revision')
        if slots is None:
            slots = torch.zeros(max_slots, max_tokens, width, dtype=torch.float32)
        if lengths is None:
            lengths = torch.zeros(max_slots, dtype=torch.long)
        if occupied is None:
            occupied = torch.zeros(max_slots, dtype=torch.bool)
        if (not isinstance(slots, torch.Tensor) or slots.shape != (max_slots, max_tokens, width)
                or not slots.is_floating_point() or not isinstance(lengths, torch.Tensor)
                or lengths.shape != (max_slots,) or lengths.dtype != torch.long
                or not isinstance(occupied, torch.Tensor) or occupied.shape != (max_slots,)
                or occupied.dtype != torch.bool or bool((lengths < 0).any())
                or bool((lengths > max_tokens).any()) or bool((occupied != (lengths > 0)).any())
                or not bool(torch.isfinite(slots).all())):
            raise ValueError('Invalid semantic phrase slot tensors')
        self.width = width
        self.max_slots = max_slots
        self.max_tokens = max_tokens
        self.slots = slots.detach().clone()
        self.lengths = lengths.detach().clone()
        self.occupied = occupied.detach().clone()
        self.revision = revision

    def validate(self):
        SemanticPhraseSlotState(self.width, self.max_slots, self.max_tokens,
                                slots=self.slots, lengths=self.lengths,
                                occupied=self.occupied, revision=self.revision)

    def put(self, index, sequence):
        if type(index) is not int or not 0 <= index < self.max_slots:
            raise ValueError('Semantic phrase slot index is outside the bank')
        if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 2
                or sequence.shape[1] != self.width or not sequence.is_floating_point()
                or not 1 <= sequence.shape[0] <= self.max_tokens
                or not bool(torch.isfinite(sequence).all())):
            raise ValueError('Semantic phrase sequence has an invalid shape or value')
        slots = self.slots.clone()
        lengths = self.lengths.clone()
        occupied = self.occupied.clone()
        slots[index].zero_()
        slots[index, :sequence.shape[0]] = sequence.detach().to(dtype=slots.dtype, device=slots.device)
        lengths[index] = sequence.shape[0]
        occupied[index] = True
        return SemanticPhraseSlotState(self.width, self.max_slots, self.max_tokens,
                                       slots=slots, lengths=lengths, occupied=occupied,
                                       revision=self.revision + 1)

    def get(self, index, *, device=None):
        if type(index) is not int or not 0 <= index < self.max_slots:
            raise ValueError('Semantic phrase slot index is outside the bank')
        if not bool(self.occupied[index]):
            raise KeyError('Semantic phrase slot is empty')
        sequence = self.slots[index, :int(self.lengths[index])].clone()
        return sequence if device is None else sequence.to(device)


def save(path, state, *, base_sha256, bridge_sha256, metadata=None):
    if not isinstance(state, SemanticPhraseSlotState) or not isinstance(base_sha256, str) or not base_sha256:
        raise ValueError('State and base binding are required')
    state.validate()
    info = {'format': FORMAT, 'width': state.width, 'max_slots': state.max_slots,
            'max_tokens': state.max_tokens, 'revision': state.revision,
            'base_sha256': base_sha256, 'bridge_sha256': bridge_sha256,
            'metadata': dict(metadata or {})}
    tensors = {'slots': state.slots, 'lengths': state.lengths, 'occupied': state.occupied}
    atomic_save({'metadata': info, 'tensors': {key: value.cpu().clone() for key, value in tensors.items()},
                 'integrity_sha256': _digest(info, tensors)}, path)
    return _digest(info, tensors)


def load(path, *, expected_base_sha256=None, expected_bridge_sha256=None, device='cpu'):
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'metadata', 'tensors', 'integrity_sha256'}:
        raise ValueError('Invalid semantic phrase slot checkpoint')
    metadata, tensors = payload['metadata'], payload['tensors']
    if (not isinstance(metadata, dict) or metadata.get('format') != FORMAT
            or set(tensors) != {'slots', 'lengths', 'occupied'}
            or _digest(metadata, tensors) != payload['integrity_sha256']):
        raise ValueError('Semantic phrase slot checkpoint integrity failure')
    if expected_base_sha256 is not None and metadata.get('base_sha256') != expected_base_sha256:
        raise ValueError('Semantic phrase slot base binding mismatch')
    if expected_bridge_sha256 is not None and metadata.get('bridge_sha256') != expected_bridge_sha256:
        raise ValueError('Semantic phrase slot bridge binding mismatch')
    state = SemanticPhraseSlotState(metadata['width'], metadata['max_slots'], metadata['max_tokens'],
                                    slots=tensors['slots'], lengths=tensors['lengths'],
                                    occupied=tensors['occupied'], revision=metadata['revision'])
    state.slots = state.slots.to(device)
    state.lengths = state.lengths.to(device)
    state.occupied = state.occupied.to(device)
    return state, metadata
