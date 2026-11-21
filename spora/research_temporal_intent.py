"""Experimental learned attention over evidence for memory time selection."""

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

    def forward(self, vectors):
        read = self.read(vectors)
        versions = self.version(vectors).log_softmax(-1)
        return torch.cat((F.logsigmoid(read) + versions, F.logsigmoid(-read)), dim=-1)


def temporal_logits(payload, features, mask):
    width = features.shape[1] * features.shape[-1]
    classes = len(payload['classes'])
    architecture = payload.get('architecture', 'mean')
    if architecture == 'mean':
        head = nn.Linear(width, classes)
        args = (question_mean(features, mask),)
    elif architecture == 'token_attention':
        head = TemporalTokenGate(width, classes)
        args = (features, mask)
    elif architecture == 'factorized':
        head = FactorizedTemporalGate(width)
        args = (question_mean(features, mask),)
    else:
        raise ValueError('Unknown temporal intent architecture')
    head.load_state_dict(payload['state'], strict=True)
    return head.eval()(*args)


def time_selection(probabilities, threshold, *, factorized=False):
    if factorized:
        read_mass = probabilities[:, :3].sum(-1)
        best_confidence, winner = probabilities[:, :3].max(-1)
        eligible = (read_mass >= threshold) & (best_confidence >= .45 * read_mass)
    else:
        confidence, winner = probabilities.max(-1)
        eligible = confidence >= threshold
    return torch.where(eligible, winner, 3)
