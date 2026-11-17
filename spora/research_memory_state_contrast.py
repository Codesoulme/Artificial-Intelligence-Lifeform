"""Research-only objective for separating answer likelihoods by memory state."""
import math

from torch.nn import functional as F
from spora.research_counterfactual_span_loss import span_mean_log_probability


def memory_state_contrast_loss(a_on_a, b_on_a, a_on_b, b_on_b, targets_a, targets_b, margin=0.25):
    """Raise each answer's score under its own state relative to the other state.

    Candidate prefixes are held constant within each comparison; the answer
    need not be naturally more probable than its competitor without memory.
    This alone cannot ensure that the correct answer is top ranked or generated.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid margin')
    aa = span_mean_log_probability(a_on_a, targets_a)
    ba = span_mean_log_probability(b_on_a, targets_b)
    ab = span_mean_log_probability(a_on_b, targets_a)
    bb = span_mean_log_probability(b_on_b, targets_b)
    if not (aa.shape == ba.shape == ab.shape == bb.shape):
        raise ValueError('Counterfactual batches must match')
    return 0.5 * (F.softplus(margin - (aa - ab)).mean()
                  + F.softplus(margin - (bb - ba)).mean())
