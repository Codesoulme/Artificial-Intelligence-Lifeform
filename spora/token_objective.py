"""One token-mean objective across differently padded source microbatches."""
import torch
from torch.nn import functional as F


def backward_token_budget(model, state, batches):
    """Caller clips/updates once. Gradients weight tokens, not source batches.

    The sampler decides when to stop collecting whole examples. Its final
    microbatch can overshoot the requested budget; use the actual denominator.
    This function does not change optimizer learning rates or perform updates.
    """
    if not batches:
        raise ValueError('At least one supervised microbatch required')
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
        if not bool(torch.isfinite(part)):raise ValueError('Nonfinite token-budget objective')
        part.backward();result+=part.detach()
        del logits,part,x,y
    return result,total
