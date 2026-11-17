"""Response-only loss for controlled causal-LM research experiments."""
import torch
import torch.nn.functional as F


def response_token_loss(logits, tokens, prompt_length):
    if (logits.ndim != 3 or tokens.ndim != 2 or tokens.dtype != torch.long
            or logits.shape[:2] != tokens.shape or type(prompt_length) is not int
            or not 1 <= prompt_length < tokens.shape[1]):
        raise ValueError('Invalid causal response inputs')
    scores=logits[:,prompt_length-1:-1,:].float()
    targets=tokens[:,prompt_length:]
    return F.cross_entropy(scores.reshape(-1,scores.shape[-1]),targets.reshape(-1))
