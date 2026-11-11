"""Experimental explicit name verification stored as tensors in an individual.

This is a deterministic identity constraint, not a learned representation and
not proof that a name belongs to a real person. Enrollment and the completeness
of an extracted name must be established by a caller before use.
"""
import hashlib
import unicodedata

import torch
from torch import nn


def _fingerprint(name):
    if not isinstance(name, str):
        raise ValueError('A name string is required')
    canonical = unicodedata.normalize('NFC', name)
    if (not canonical or len(canonical) > 128 or canonical != canonical.strip()
            or any(unicodedata.category(char).startswith('C') for char in canonical)):
        raise ValueError('A bounded, nonempty name without control characters is required')
    # Domain separation leaves room for other kinds of fingerprints later.
    return list(hashlib.sha256(b'spora-explicit-name-v1\0' + canonical.encode('utf-8')).digest())


class IdentityFingerprints(nn.Module):
    """Fixed-capacity exact-name bindings; explicitly enrolled key slots only."""

    def __init__(self, key_count):
        super().__init__()
        if type(key_count) is not int or not 1 <= key_count <= 65536:
            raise ValueError('Invalid identity key count')
        self.key_count = key_count
        # int32 is supported by existing durable tensor storage on every device.
        self.register_buffer('digests', torch.zeros(key_count, 32, dtype=torch.int32))
        self.register_buffer('occupied', torch.zeros(key_count, dtype=torch.bool))

    def config_dict(self):
        return {'key_count': self.key_count}

    def validate(self, model=None):
        if (type(self.key_count) is not int or not 1 <= self.key_count <= 65536
                or self.digests.shape != (self.key_count, 32) or self.digests.dtype != torch.int32
                or self.occupied.shape != (self.key_count,) or self.occupied.dtype != torch.bool
                or self.digests.device != self.occupied.device
                or bool(((self.digests < 0) | (self.digests > 255)).any())):
            raise ValueError('Invalid identity fingerprint state')
        if model is not None and model.config.key_count != self.key_count:
            raise ValueError('Identity key count does not match individual')
        if bool((self.digests[~self.occupied] != 0).any()):
            raise ValueError('Unoccupied identity slot contains a fingerprint')
        active = self.digests[self.occupied]
        if len(active) and torch.unique(active, dim=0).shape[0] != len(active):
            raise ValueError('Duplicate identity fingerprint')
        return self

    @torch.no_grad()
    def bind(self, key, name):
        """Bind an unoccupied slot; reject reassignment or duplicate names."""
        self.validate()
        if type(key) is not int or not 0 <= key < self.key_count:
            raise ValueError('Identity key out of range')
        digest = torch.tensor(_fingerprint(name), dtype=torch.int32, device=self.digests.device)
        if bool(self.occupied[key]):
            raise ValueError('Identity slot is already occupied')
        if bool(((self.digests == digest).all(-1) & self.occupied).any()):
            raise ValueError('Name is already bound to an identity slot')
        self.digests[key].copy_(digest)
        self.occupied[key] = True

    @torch.no_grad()
    def lookup(self, name):
        """Return the matching key or None; never allocate a slot on lookup."""
        self.validate()
        digest = torch.tensor(_fingerprint(name), dtype=torch.int32, device=self.digests.device)
        matching = ((self.digests == digest).all(-1) & self.occupied).nonzero(as_tuple=True)[0]
        return int(matching[0]) if len(matching) == 1 else None
