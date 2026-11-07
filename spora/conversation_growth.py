"""Function-preserving width doubling for the local conversation path.

The frozen individual is reused. No external weights and no vocabulary changes.
Balanced column perturbations cancel on replicated inputs while allowing the
replicas to learn differently after the first update. Final RMS scale is halved
because the input/output embedding is tied.
"""
from dataclasses import replace
import torch
from .conversation import ConversationBridge
from .language_training import interface_state


def widen_conversation(source, *, seed=231301, noise_std=0.001):
    if noise_std < 0 or not torch.isfinite(torch.tensor(noise_std)):
        raise ValueError('Finite nonnegative perturbation required')
    c = source.config
    device = next(source.parameters()).device
    config = replace(c, width=c.width * 2, heads=c.heads * 2,
                     hidden=c.hidden * 2)
    # Construction initializes only the new path, never the frozen organism.
    target = ConversationBridge(source.organism, config).to(device).eval()
    old = interface_state(source)
    new = interface_state(target)
    rng = torch.Generator(device='cpu').manual_seed(seed)

    def columns(weight):
        base = torch.cat((weight / 2, weight / 2), dim=1)
        if noise_std:
            noise = torch.randn(weight.shape, generator=rng,
                                dtype=weight.dtype) * noise_std
            noise = noise.to(weight.device)
            base += torch.cat((noise, -noise), dim=1)
        return base

    for name, value in old.items():
        if name == 'embedding.weight':
            mapped = torch.cat((value, value), dim=1)
        elif name == 'norm.weight':
            mapped = torch.cat((value / 2, value / 2))
        elif name in ('memory_gate', 'typed_memory.page_keys'):
            mapped = value.clone()
        elif name.startswith('memory_query.'):
            mapped = columns(value)
        elif name.startswith('memory_out.'):
            mapped = torch.cat((value, value), dim=0)
        elif name.endswith(('norm1.weight', 'norm2.weight')):
            mapped = torch.cat((value, value))
        elif name.endswith('qkv.weight'):
            # Keep Q/K/V groups separate; retain rotary head width.
            mapped = columns(torch.cat([
                torch.cat((group, group), dim=0)
                for group in value.chunk(3, dim=0)
            ], dim=0))
        elif name.endswith('up.weight'):
            # Keep SwiGLU gate and value groups separate.
            mapped = columns(torch.cat([
                torch.cat((group, group), dim=0)
                for group in value.chunk(2, dim=0)
            ], dim=0))
        elif name.endswith(('out.weight', 'down.weight')):
            mapped = columns(torch.cat((value, value), dim=0))
        else:
            raise ValueError('Unsupported language tensor: ' + name)
        if mapped.shape != new[name].shape:
            raise ValueError('Growth shape mismatch: ' + name)
        new[name] = mapped
    if old.keys() != new.keys():
        raise ValueError('Growth must cover every language tensor')
    loaded = target.load_state_dict(new, strict=False)
    if loaded.unexpected_keys or any(not k.startswith('organism.')
                                     for k in loaded.missing_keys):
        raise ValueError('Incomplete language restoration during growth')
    return target
