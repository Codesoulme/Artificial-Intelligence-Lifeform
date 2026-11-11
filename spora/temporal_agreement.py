"""Experimental agreement gate for independent temporal classifier heads."""
import math

import torch
from torch import nn


class TemporalAgreement(nn.Module):
    """Return baseline logits only when both heads choose the same read class.

    Disagreement and either head's abstention return an abstention logit. This
    is a conservative heuristic, not a calibrated error guarantee.
    """

    def __init__(self, baseline, candidate, baseline_threshold, candidate_threshold):
        super().__init__()
        if not isinstance(baseline, nn.Module) or not isinstance(candidate, nn.Module):
            raise ValueError('Two classifier modules required')
        for value in (baseline_threshold, candidate_threshold):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Thresholds must be finite probabilities')
        self.baseline, self.candidate = baseline, candidate
        self.baseline_threshold = float(baseline_threshold)
        self.candidate_threshold = float(candidate_threshold)

    def forward(self, vectors):
        left, right = self.baseline(vectors), self.candidate(vectors)
        if (left.ndim != 2 or left.shape[-1] != 4 or right.shape != left.shape
                or not left.is_floating_point() or not right.is_floating_point()
                or left.device != right.device
                or not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all())):
            raise ValueError('Finite four-class temporal logits required')
        lp, rp = left.softmax(-1), right.softmax(-1)
        lc, rc = lp.argmax(-1), rp.argmax(-1)
        agree = ((lc == rc) & (lc != 3) & (lp[:, :3].max(-1).values >= self.baseline_threshold)
                 & (rp[:, :3].max(-1).values >= self.candidate_threshold))
        abstain = torch.full_like(left, -10000.)
        abstain[:, 3] = 0.
        return torch.where(agree[:, None], left, abstain)
