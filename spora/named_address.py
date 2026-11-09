"""Experimental name-to-address projection stored with a SPORA individual.

The caller must supply the matching frozen encoder's question-pooled features
and honestly computed fingerprints. Legacy bindings require explicit opt-in.
This component does not extract text, write memory, or establish open-world
entity identity; its acceptance threshold is an uncalibrated experiment.
"""
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def address_basis_sha256(model):
    if len(model.memories) != 2 or any(not hasattr(site, 'current_basis') for site in model.memories):
        raise ValueError('Two address bases required')
    digest = hashlib.sha256()
    for site in model.memories:
        basis = site.current_basis.detach().cpu().contiguous()
        if basis.dtype != torch.float32 or basis.ndim != 2 or not bool(torch.isfinite(basis).all()):
            raise ValueError('Expected finite float32 address basis')
        digest.update(json.dumps(list(basis.shape), separators=(',', ':')).encode('ascii'))
        digest.update(basis.numpy().tobytes())
    return digest.hexdigest()


def feature_files_sha256(model_dir):
    """Fingerprint configuration and tokenizer files, without loading executable code."""
    entries = []
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        digest = hashlib.sha256()
        with (Path(model_dir) / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        entries.append([name, digest.hexdigest()])
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode('ascii')).hexdigest()


class NamedAddressBinding(nn.Module):
    def __init__(self, feature_width, memory_width, feature_model_sha256, threshold,
                 feature_layers=(6, 18), feature_contract='question-mean-v1',
                 basis_sha256=None, feature_config_sha256=None):
        super().__init__()
        self._validate_config(feature_width, memory_width, feature_model_sha256, threshold,
                              feature_layers, feature_contract, basis_sha256, feature_config_sha256)
        self.feature_width = feature_width
        self.memory_width = memory_width
        self.feature_model_sha256 = feature_model_sha256
        self.threshold = float(threshold)
        self.feature_layers = tuple(feature_layers)
        self.feature_contract = feature_contract
        self.basis_sha256 = basis_sha256
        self.feature_config_sha256 = feature_config_sha256
        self.register_buffer('weights', torch.zeros(2, feature_width, memory_width))

    @staticmethod
    def _validate_config(feature_width, memory_width, feature_model_sha256, threshold,
                         feature_layers, feature_contract, basis_sha256=None, feature_config_sha256=None):
        if (type(feature_width) is not int or not 1 <= feature_width <= 16384
                or type(memory_width) is not int or not 1 <= memory_width <= 4096):
            raise ValueError('Invalid binding widths')
        if not isinstance(feature_model_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', feature_model_sha256):
            raise ValueError('Expected lowercase encoder SHA-256')
        if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError('Threshold must be finite and in [0, 1]')
        if (not isinstance(feature_layers, (list, tuple)) or len(feature_layers) != 2
                or any(type(i) is not int or i <= 0 for i in feature_layers)
                or feature_layers[0] == feature_layers[1] or feature_contract != 'question-mean-v1'):
            raise ValueError('Unsupported feature contract')
        if (basis_sha256 is None) != (feature_config_sha256 is None):
            raise ValueError('Both binding fingerprints are required together')
        if basis_sha256 is not None and any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                                            for value in (basis_sha256, feature_config_sha256)):
            raise ValueError('Invalid binding fingerprint')

    def config_dict(self):
        result = {'feature_width': self.feature_width, 'memory_width': self.memory_width,
                'feature_model_sha256': self.feature_model_sha256, 'threshold': self.threshold,
                'feature_layers': list(self.feature_layers), 'feature_contract': self.feature_contract}
        if self.basis_sha256 is not None:
            result.update(basis_sha256=self.basis_sha256, feature_config_sha256=self.feature_config_sha256)
        return result

    def validate(self, model):
        # Revalidate mutable metadata before it enters a saved schema.
        self._validate_config(**self.config_dict())
        if (len(model.memories) != 2 or model.config.current_address != 'orthogonal'
                or model.config.memory_width != self.memory_width
                or any(not hasattr(site, 'current_basis') for site in model.memories)
                or self.weights.shape != (2, self.feature_width, self.memory_width)
                or not bool(torch.isfinite(self.weights).all())):
            raise ValueError('Binding does not match individual')
        if self.basis_sha256 is not None and address_basis_sha256(model) != self.basis_sha256:
            raise ValueError('Address basis fingerprint mismatch')

    def forward(self, features, model, feature_model_sha256, feature_config_sha256=None, *, allow_legacy_unbound=False):
        if feature_model_sha256 != self.feature_model_sha256:
            raise ValueError('Feature encoder mismatch')
        if type(allow_legacy_unbound) is not bool:
            raise ValueError('Legacy permission must be boolean')
        if self.basis_sha256 is None:
            if not allow_legacy_unbound:
                raise ValueError('Legacy binding lacks fingerprints')
        elif feature_config_sha256 != self.feature_config_sha256:
            raise ValueError('Feature configuration mismatch')
        if getattr(model, 'named_address_binding', None) is not self:
            raise ValueError('Binding must belong to the queried individual')
        if (not isinstance(features, torch.Tensor) or features.ndim != 3
                or features.shape[1:] != (2, self.feature_width) or not features.shape[0]
                or not features.is_floating_point() or not bool(torch.isfinite(features).all())):
            raise ValueError('Expected finite [batch, 2, feature_width] features')
        self.validate(model)
        features = features.to(device=self.weights.device, dtype=self.weights.dtype)
        features = F.normalize(features, dim=-1)
        raw = torch.einsum('bsh,shd->bsd', features, self.weights)
        queries = F.normalize(raw, dim=-1)
        bases = torch.stack([F.normalize(site.current_basis.to(self.weights), dim=-1) for site in model.memories])
        scores, indices = torch.einsum('bsd,skd->bsk', queries, bases).max(-1)
        accepted = ((scores >= self.threshold).all(-1) & (indices[:, 0] == indices[:, 1])
                    & (raw.norm(dim=-1) > 1e-12).all(-1))
        return {'indices': indices, 'scores': scores, 'accepted': accepted}
