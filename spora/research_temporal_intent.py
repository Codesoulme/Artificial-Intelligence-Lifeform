"""Experimental learned attention over evidence for memory time selection."""

import math
import unicodedata
import zlib

import torch
from torch import nn
from torch.nn import functional as F


def question_mean(features, mask):
    weights = mask[:, None, :, None].float()
    masked = features.float().masked_fill(~mask[:, None, :, None], 0.)
    means = (masked * weights).sum(2) / weights.sum(2).clamp_min(1)
    return F.layer_norm(means, (means.shape[-1],)).flatten(1)


class TemporalTokenGate(nn.Module):
    """Combine a whole-question decision with class-specific token evidence."""

    def __init__(self, feature_width, classes=4):
        super().__init__()
        self.mean = nn.Linear(feature_width, classes)
        self.project = nn.Linear(feature_width, 48)
        self.attention = nn.Linear(48, classes)
        self.evidence = nn.Linear(48, classes)

    def forward(self, features, mask):
        normalized = F.layer_norm(features.float(), (features.shape[-1],))
        tokens = normalized.transpose(1, 2).flatten(2).masked_fill(~mask[..., None], 0.)
        hidden = F.gelu(self.project(tokens))
        attention = self.attention(hidden).masked_fill(~mask[..., None], -torch.inf).softmax(dim=1)
        return self.mean(question_mean(features, mask)) + (attention * self.evidence(hidden)).sum(dim=1)


class FactorizedTemporalGate(nn.Module):
    """Estimate whether to read and which version to read independently."""

    def __init__(self, feature_width):
        super().__init__()
        self.read = nn.Linear(feature_width, 1)
        self.version = nn.Linear(feature_width, 3)

    def forward(self, vectors, version_delta=None):
        read = self.read(vectors)
        versions = self.version(vectors)
        if version_delta is not None:
            versions = versions + version_delta
        versions = versions.log_softmax(-1)
        return torch.cat((F.logsigmoid(read) + versions, F.logsigmoid(-read)), dim=-1)


def temporal_character_features(texts, width):
    if type(width) is not int or width < 1 or any(not isinstance(text, str) for text in texts):
        raise ValueError('Expected character feature width and question strings')
    counts = torch.zeros((len(texts), width), dtype=torch.float32)
    for row, text in enumerate(texts):
        chars = ''.join(unicodedata.normalize('NFKC', text).split())
        for length in (1, 2, 3):
            for start in range(len(chars) - length + 1):
                encoded = chars[start:start + length].encode('utf-8')
                counts[row, zlib.crc32(encoded) % width] += 1.
    return F.normalize(counts, dim=-1) * math.sqrt(width)


def temporal_logits(payload, features, mask, *, texts=None):
    width = features.shape[1] * features.shape[-1]
    classes = len(payload['classes'])
    architecture = payload.get('architecture', 'mean')
    residual = payload.get('version_residual')
    selector = payload.get('version_selector')
    if selector is not None and (architecture != 'factorized' or residual is not None):
        raise ValueError('Version selector requires an unmodified factorized head')
    if residual is not None and (architecture != 'factorized' or not isinstance(residual, dict)):
        raise ValueError('Version residual requires a factorized temporal head')
    if architecture == 'mean':
        head = nn.Linear(width, classes)
        args = (question_mean(features, mask),)
    elif architecture == 'token_attention':
        head = TemporalTokenGate(width, classes)
        args = (features, mask)
    elif architecture == 'factorized':
        head = FactorizedTemporalGate(width)
        vectors = question_mean(features, mask)
        args = (vectors,)
        if residual is not None:
            char_width = residual['char_width']
            strength = residual['strength']
            if (type(char_width) is not int or not 0 <= char_width <= 4096
                    or type(strength) not in (int, float) or not 0. <= strength <= 2.):
                raise ValueError('Invalid temporal version residual configuration')
            inputs = vectors
            if char_width:
                if texts is None or len(texts) != len(vectors):
                    raise ValueError('Temporal lexical version head needs one question per feature row')
                lexical = temporal_character_features(texts, char_width).to(vectors)
                inputs = torch.cat((vectors, lexical), dim=-1)
            weight = residual['weight']
            if (not isinstance(weight, torch.Tensor) or weight.shape != (3, inputs.shape[-1])
                    or not weight.is_floating_point() or not bool(torch.isfinite(weight).all())):
                raise ValueError('Invalid temporal version residual projection')
            delta = F.linear(inputs, weight.to(vectors))
            args = (vectors, strength * delta)
    else:
        raise ValueError('Unknown temporal intent architecture')
    head.load_state_dict(payload['state'], strict=True)
    logits = head.eval()(*args)
    if selector is not None:
        char_width = selector['char_width']
        confidence = selector['threshold']
        if (type(char_width) is not int or not 0 <= char_width <= 4096
                or type(confidence) not in (int, float) or not .5 <= confidence <= .95):
            raise ValueError('Invalid guarded version selector configuration')
        inputs = vectors
        if char_width:
            if texts is None or len(texts) != len(vectors):
                raise ValueError('Version selector requires one question per feature row')
            inputs = torch.cat((vectors, temporal_character_features(texts, char_width).to(vectors)), dim=-1)
        weight = selector['weight']
        if (not isinstance(weight, torch.Tensor) or weight.shape != (3, inputs.shape[-1])
                or not weight.is_floating_point() or not bool(torch.isfinite(weight).all())):
            raise ValueError('Invalid guarded version selector weights')
        delta = F.linear(inputs, weight.to(vectors))
        adjusted = head(vectors, delta)
        applies = ((time_selection(logits.softmax(-1), payload['threshold'], factorized=True) != 3)
                   & ((head.version(vectors) + delta).softmax(-1).amax(-1) >= confidence))
        return torch.where(applies[:, None], adjusted, logits)
    override = None if residual is None else residual.get('past_override')
    if override is not None:
        if type(override) not in (int, float) or not .55 <= override <= .95:
            raise ValueError('Invalid past-version correction confidence')
        original = head(vectors)
        choose_past = past_override_mask(original.softmax(-1), logits.softmax(-1), payload['threshold'], override)
        logits = torch.where(choose_past[:, None], logits, original)
    return logits


def past_override_mask(original, revised, threshold, confidence):
    current_decision = time_selection(original, threshold, factorized=True)
    original_past = original[:, 1] / original[:, :3].sum(-1).clamp_min(1e-8)
    revised_past = revised[:, 1] / revised[:, :3].sum(-1).clamp_min(1e-8)
    return ((current_decision == 0) | (current_decision == 2)) & (revised_past >= confidence) & (
        revised_past >= original_past + .15
    )


def time_selection(probabilities, threshold, *, factorized=False):
    if factorized:
        read_mass = probabilities[:, :3].sum(-1)
        best_confidence, winner = probabilities[:, :3].max(-1)
        eligible = (read_mass >= threshold) & (best_confidence >= .45 * read_mass)
    else:
        confidence, winner = probabilities.max(-1)
        eligible = confidence >= threshold
    return torch.where(eligible, winner, 3)
