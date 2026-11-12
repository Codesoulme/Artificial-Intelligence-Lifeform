"""Versioned polarity and value heads for natural-language memory writes."""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .integrated_checkpoint import replacement_feature_variant_sha256
from .named_address import feature_files_sha256
from .structure_transfer import SPORALayerReplacement


class ValueSpanHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.accept = nn.Linear(width, 1)
        self.boundary = nn.Linear(width, 2)

    def forward(self, vectors):
        pooled = vectors.mean((1, 2))
        return self.accept(pooled).squeeze(-1), self.boundary(vectors).mean(1)


class NaturalWriteHeadBundle(nn.Module):
    """Immutable research heads bound to one exact base/feature provenance."""

    FORMAT = 'spora-natural-write-coverage-heads-v1'

    def __init__(self, polarity, value, polarity_threshold, value_threshold,
                 *, artifact_sha256, base_sha256, checkpoint_sha256, feature_variant_sha256):
        super().__init__()
        if not isinstance(polarity, nn.Module) or not isinstance(value, nn.Module):
            raise ValueError('Natural write heads must be modules')
        for name, threshold in (('polarity', polarity_threshold), ('value', value_threshold)):
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                raise ValueError(f'Invalid {name} threshold')
        self.polarity = polarity.eval()
        self.value = value.eval()
        self.polarity_threshold = float(polarity_threshold)
        self.value_threshold = float(value_threshold)
        self.artifact_sha256 = artifact_sha256
        self.base_sha256 = base_sha256
        self.checkpoint_sha256 = checkpoint_sha256
        self.feature_variant_sha256 = feature_variant_sha256
        self.requires_grad_(False)

    def config_dict(self):
        return {'format': self.FORMAT, 'polarity_threshold': self.polarity_threshold,
                'value_threshold': self.value_threshold,
                'base_sha256': self.base_sha256,
                'checkpoint_sha256': self.checkpoint_sha256,
                'feature_variant_sha256': self.feature_variant_sha256}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_natural_write_heads(path, base, model_dir, checkpoint_path, *, device=None):
    """Load and provenance-check a natural-write head bundle."""
    path = Path(path).resolve()
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or data.get('format') != NaturalWriteHeadBundle.FORMAT:
        raise ValueError('Unsupported natural write head bundle')
    artifact_sha = _sha(path)
    if not isinstance(data.get('polarity'), dict) or not isinstance(data.get('value'), dict):
        raise ValueError('Natural write head states are missing')
    base_sha = _sha(Path(model_dir) / 'model.safetensors')
    checkpoint_sha = _sha(checkpoint_path)
    replacements = {i: layer for i, layer in enumerate(base.model.layers)
                    if isinstance(layer, SPORALayerReplacement)}
    variant = replacement_feature_variant_sha256(model_dir, replacements)
    for field, actual in (('base_sha256', base_sha), ('source_checkpoint_sha256', checkpoint_sha),
                          ('feature_variant_sha256', variant)):
        if data.get(field) != actual:
            raise ValueError(f'Natural write head {field} mismatch')
    hidden = getattr(base.config, 'hidden_size', None)
    if type(hidden) is not int or hidden <= 0:
        raise ValueError('Base hidden size is invalid')
    polarity = nn.Linear(hidden * 2, 6)
    value = ValueSpanHead(hidden)
    try:
        polarity.load_state_dict(data['polarity'], strict=True)
        value.load_state_dict(data['value'], strict=True)
    except RuntimeError as error:
        raise ValueError('Natural write head tensor shapes do not match the base') from error
    bundle = NaturalWriteHeadBundle(
        polarity, value, data.get('polarity_threshold'), data.get('value_threshold'),
        artifact_sha256=artifact_sha, base_sha256=base_sha,
        checkpoint_sha256=checkpoint_sha, feature_variant_sha256=variant)
    return bundle.to(device if device is not None else next(base.parameters()).device).eval()
