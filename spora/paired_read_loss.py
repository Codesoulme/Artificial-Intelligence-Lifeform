"""A contrastive objective for the same query under two memory states."""
import torch
from torch.nn import functional as F


def paired_read_loss(first_margin, second_margin, margin=1.0):
    """Prefer the first value over the second more strongly in the first state."""
    if (not isinstance(first_margin, torch.Tensor) or not isinstance(second_margin, torch.Tensor)
            or first_margin.shape != second_margin.shape or not first_margin.is_floating_point()
            or not second_margin.is_floating_point() or isinstance(margin, bool)
            or not isinstance(margin, (int, float)) or not 0 < margin <= 100):
        raise ValueError('Invalid paired margins')
    return F.softplus(margin - (first_margin - second_margin)).mean()
