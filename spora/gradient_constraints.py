"""Gradient projection for experimental replay-based training."""
import torch


def project_against_reference(gradients, references):
    """Project one gradient onto the reference's nonnegative half-space.

    A zero reference leaves the gradient unchanged. Inputs are never mutated.
    This first-order constraint does not guarantee that a finite optimizer step
    preserves old loss or model behavior.
    """
    if (not isinstance(gradients, (list, tuple)) or not isinstance(references, (list, tuple))
            or not gradients or len(gradients) != len(references)):
        raise ValueError('Matching nonempty gradient sequences required')
    for g, r in zip(gradients, references):
        if (not isinstance(g, torch.Tensor) or not isinstance(r, torch.Tensor)
                or g.shape != r.shape or g.device != r.device or g.dtype != r.dtype
                or not g.is_floating_point() or not bool(torch.isfinite(g).all())
                or not bool(torch.isfinite(r).all())):
            raise ValueError('Matching finite floating-point gradients required')
    dot = sum((g.double() * r.double()).sum() for g, r in zip(gradients, references))
    norm = sum(r.double().square().sum() for r in references)
    if not bool(torch.isfinite(dot)) or not bool(torch.isfinite(norm)):
        raise ValueError('Gradient projection overflow')
    if float(dot) >= 0.0 or float(norm) == 0.0:
        return [g.detach().clone() for g in gradients], False
    factor = dot / norm
    return [(g.double() - factor * r.double()).to(g.dtype).detach() for g, r in zip(gradients, references)], True
