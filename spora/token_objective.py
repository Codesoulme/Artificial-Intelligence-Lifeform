"""One token-mean objective across differently padded source microbatches."""
import torch
from torch.nn import functional as F


def _repeated_ngram_targets(tokens, labels, ngram=4):
    """Mark target positions whose n-gram already appeared earlier in the row."""
    mask = torch.zeros_like(labels, dtype=torch.bool)
    if ngram < 2:
        raise ValueError('N-gram length must be at least two')
    for row in range(tokens.shape[0]):
        seen = set()
        values = tokens[row].tolist()
        history = ngram - 1
        for index, target in enumerate(labels[row].tolist()):
            if target == -100 or index + 1 < history:
                continue
            gram = tuple(values[index + 1 - history:index + 1]) + (target,)
            if gram in seen:
                mask[row, index] = True
            seen.add(gram)
    return mask


def backward_token_budget(model, state, batches, unlikelihood_weight=0.0):
    """Caller clips/updates once. Gradients weight tokens, not source batches.

    The sampler decides when to stop collecting whole examples. Its final
    microbatch can overshoot the requested budget; use the actual denominator.
    This function does not change optimizer learning rates or perform updates.
    """
    if not batches:
        raise ValueError('At least one supervised microbatch required')
    if not isinstance(unlikelihood_weight, (int, float)) or unlikelihood_weight < 0:
        raise ValueError('Unlikelihood weight must be nonnegative')
    total=0
    for tokens,labels in batches:
        if tokens.shape!=labels.shape or tokens.ndim!=2 or tokens.dtype!=torch.long or labels.dtype!=torch.long:
            raise ValueError('Matching integer tokens and labels required')
        total+=int((labels!=-100).sum())
    if total==0:raise ValueError('No supervised tokens in update')
    device=next(model.parameters()).device
    result=torch.zeros((),device=device)
    for tokens,labels in batches:
        if not bool((labels!=-100).any()):continue
        x=tokens.to(device);y=labels.to(device)
        logits=model(x,state)
        part=F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum')/total
        if unlikelihood_weight:
            repeated = _repeated_ngram_targets(tokens, labels)
            if bool(repeated.any()):
                probabilities = logits.softmax(-1).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                penalty = -torch.log1p(-probabilities.clamp(max=1 - 1e-5))[repeated].sum() / total
                part = part + unlikelihood_weight * penalty
        if not bool(torch.isfinite(part)):raise ValueError('Nonfinite token-budget objective')
        part.backward();result+=part.detach()
        del logits,part,x,y
    return result,total
