"""Explicit, reproducible sampling; no answer repair or repetition masking."""
import torch

def nucleus_probabilities(scores,temperature,top_p):
    if not 0<temperature<=2 or not 0<top_p<=1:raise ValueError('Invalid sampling parameters')
    ranked,indices=torch.sort(scores.float()/temperature,descending=True)
    probabilities=ranked.softmax(-1)
    # Keep the token which crosses the boundary, and always the first token.
    reject=probabilities.cumsum(-1)-probabilities>=top_p
    probabilities=probabilities.masked_fill(reject,0)
    probabilities=probabilities/probabilities.sum()
    return torch.zeros_like(probabilities).scatter(0,indices,probabilities)

@torch.no_grad()
def generate_sampled(model,state,prefix,eos,budget=128,forbidden=(),temperature=.7,top_p=.9,seed=0):
    if not prefix or len(prefix)+budget>model.config.context:raise ValueError('Invalid prompt/budget')
    device=next(model.parameters()).device;rng=torch.Generator(device=device).manual_seed(seed)
    tokens=torch.tensor([prefix],device=device);past=None;output=[];ended=False
    for _ in range(budget):
        logits,past=model(tokens,state,past=past,use_past=True);scores=logits[0,-1].clone()
        if not bool(torch.isfinite(scores).all()):raise ValueError('Nonfinite generation')
        if forbidden:scores[list(forbidden)]=-torch.inf
        probabilities=nucleus_probabilities(scores,temperature,top_p)
        token=int(torch.multinomial(probabilities,1,generator=rng))
        if token==eos:ended=True;break
        output.append(token);tokens=torch.tensor([[token]],device=device)
    return dict(tokens=output,ended_with_eos=ended)
