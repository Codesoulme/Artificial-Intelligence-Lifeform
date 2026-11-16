"""Research-only locality gate for a low-rank writer correction."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def calibrate_threshold(anchor, negatives):
    if (not isinstance(anchor, torch.Tensor) or anchor.ndim != 1
            or not isinstance(negatives, torch.Tensor) or negatives.ndim != 2
            or negatives.shape[1] != anchor.numel() or not negatives.shape[0]
            or not bool(torch.isfinite(anchor).all()) or not bool(torch.isfinite(negatives).all())
            or float(anchor.norm()) == 0 or bool((negatives.norm(dim=-1) == 0).any())):
        raise ValueError('Expected finite nonzero anchor and negative features')
    scores = F.normalize(negatives.float(), dim=-1) @ F.normalize(anchor.float(), dim=0)
    maximum = float(scores.max())
    if maximum >= 1 - 1e-6:
        raise ValueError('Training negative is indistinguishable from the anchor')
    return (1. + maximum) / 2.


class LocalWriter(nn.Module):
    """Select a correction using feature proximity, with a frozen base map."""

    def __init__(self, original_weight, corrected_weight, anchor, threshold):
        super().__init__()
        if (not isinstance(original_weight, torch.Tensor) or original_weight.ndim != 2
                or original_weight.shape[0] != original_weight.shape[1]
                or not isinstance(corrected_weight, torch.Tensor) or corrected_weight.shape != original_weight.shape
                or not isinstance(anchor, torch.Tensor) or anchor.shape != (original_weight.shape[1],)
                or not all(bool(torch.isfinite(x).all()) for x in (original_weight, corrected_weight, anchor))
                or float(anchor.norm()) == 0 or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not -1 <= threshold <= 1):
            raise ValueError('Invalid local writer configuration')
        self.register_buffer('original_weight', original_weight.detach().float().clone())
        self.register_buffer('delta_weight', corrected_weight.detach().float() - original_weight.detach().float())
        self.register_buffer('anchor', F.normalize(anchor.detach().float(), dim=0))
        self.threshold = float(threshold)

    def active(self, hidden):
        if (not isinstance(hidden, torch.Tensor) or hidden.shape[-1] != self.anchor.numel()
                or not bool(torch.isfinite(hidden).all())):
            raise ValueError('Invalid writer feature')
        return (F.normalize(hidden.float(), dim=-1) * self.anchor).sum(-1) > self.threshold

    def forward(self, hidden):
        gate = self.active(hidden).unsqueeze(-1)
        original = F.linear(hidden.float(), self.original_weight)
        return original + gate * F.linear(hidden.float(), self.delta_weight)


class LocalWriterBank(nn.Module):
    """Apply one matching correction; ambiguous matches fall back to the base."""

    def __init__(self, original_weight, corrected_weights, anchors, thresholds):
        super().__init__()
        if (not isinstance(original_weight, torch.Tensor) or original_weight.ndim != 2
                or original_weight.shape[0] != original_weight.shape[1]
                or not isinstance(corrected_weights, (tuple, list)) or not corrected_weights
                or not isinstance(anchors, (tuple, list)) or not isinstance(thresholds, (tuple, list))
                or len(corrected_weights) != len(anchors) or len(anchors) != len(thresholds)):
            raise ValueError('Expected matching nonempty correction configurations')
        modules = [LocalWriter(original_weight, weight, anchor, threshold)
                   for weight, anchor, threshold in zip(corrected_weights, anchors, thresholds)]
        self.register_buffer('original_weight', original_weight.detach().float().clone())
        self.register_buffer('delta_weights', torch.stack([module.delta_weight for module in modules]))
        self.register_buffer('anchors', torch.stack([module.anchor for module in modules]))
        self.register_buffer('thresholds', torch.tensor(thresholds, dtype=torch.float32))

    def matches(self, hidden):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim < 1
                or hidden.shape[-1] != self.anchors.shape[1] or not bool(torch.isfinite(hidden).all())):
            raise ValueError('Invalid writer feature')
        return F.normalize(hidden.float(), dim=-1) @ self.anchors.T > self.thresholds

    def forward(self, hidden):
        matches = self.matches(hidden)
        selected = matches & (matches.sum(dim=-1, keepdim=True) == 1)
        result = F.linear(hidden.float(), self.original_weight)
        for index, delta in enumerate(self.delta_weights):
            result = result + selected[..., index:index + 1] * F.linear(hidden.float(), delta)
        return result


class FactorizedLocalWriterBank(nn.Module):
    """Apply gated low-rank corrections without expanding them to dense maps."""

    def __init__(self, original_weight, down, up, anchors, thresholds):
        super().__init__()
        if (not isinstance(original_weight, torch.Tensor) or original_weight.ndim != 2
                or original_weight.shape[0] != original_weight.shape[1]
                or not isinstance(down, torch.Tensor) or down.ndim != 3
                or not isinstance(up, torch.Tensor) or up.ndim != 3
                or down.shape[0] < 1 or down.shape[0] != up.shape[0]
                or down.shape[2] != original_weight.shape[1] or up.shape[1] != original_weight.shape[0]
                or down.shape[1] != up.shape[2] or down.shape[1] < 1
                or not isinstance(anchors, torch.Tensor) or anchors.shape != (down.shape[0], down.shape[2])
                or not isinstance(thresholds, torch.Tensor) or thresholds.shape != (down.shape[0],)
                or not all(bool(torch.isfinite(x).all()) for x in (original_weight, down, up, anchors, thresholds))
                or bool((anchors.norm(dim=-1) == 0).any())
                or bool(((thresholds < -1) | (thresholds > 1)).any())):
            raise ValueError('Invalid factorized local writer configuration')
        self.register_buffer('original_weight', original_weight.detach().float().clone())
        self.register_buffer('down', down.detach().float().clone())
        self.register_buffer('up', up.detach().float().clone())
        self.register_buffer('anchors', F.normalize(anchors.detach().float(), dim=-1))
        self.register_buffer('thresholds', thresholds.detach().float().clone())

    def matches(self, hidden):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim < 1
                or hidden.shape[-1] != self.anchors.shape[1] or not bool(torch.isfinite(hidden).all())):
            raise ValueError('Invalid writer feature')
        return F.normalize(hidden.float(), dim=-1) @ self.anchors.T > self.thresholds

    def forward(self, hidden):
        matches = self.matches(hidden)
        selected = matches & (matches.sum(dim=-1, keepdim=True) == 1)
        result = F.linear(hidden.float(), self.original_weight)
        for index in range(self.down.shape[0]):
            update = F.linear(F.linear(hidden.float(), self.down[index]), self.up[index])
            result = result + selected[..., index:index + 1] * update
        return result
