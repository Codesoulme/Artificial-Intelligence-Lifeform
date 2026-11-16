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
