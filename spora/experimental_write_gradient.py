"""Research-only target gradient for one controlled current-memory delta write.

This does not alter durable writes or the decoder's immutable snapshots. It
adds an analytically equivalent first-order path to a native reader's read_out
while retaining the actual detached matrix in the forward pass. It is valid
only for one batch row, fixed write/read addresses, one delta write and one
current page. Never apply it to history, RLS, multiple writes or production.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def controlled_write_target_gradient(reader, memory_sites, target, write_addresses, read_addresses, gates):
    """Attach the exact target Jacobian for the stated, restrictive setup."""
    if (target.ndim != 2 or target.shape[0] != 1 or not target.requires_grad
            or len(memory_sites) != 2 or len(write_addresses) != 2
            or len(read_addresses) != 2 or len(gates) != 2):
        raise ValueError('Expected one differentiable target and two fixed sites')
    handles = []
    try:
        for site_index, (site, write, read, gate) in enumerate(zip(memory_sites, write_addresses, read_addresses, gates)):
            if (write.shape != target.shape or read.shape != target.shape or gate.shape != (1,)
                    or write.requires_grad or read.requires_grad or gate.requires_grad):
                raise ValueError('Addresses and gates must be fixed and match the target')
            # NativeSPORAMemoryReader normalizes the supplied read address.
            coefficient = (gate.detach() * torch.sigmoid(site.log_lr.detach())
                           * (write.detach() * F.normalize(read.detach(), dim=-1)).sum(-1)).detach()
            native = reader.native_sites[site_index]
            def inject(module, args, *, coefficient=coefficient):
                signal, = args
                if signal.ndim != 2 or signal.shape[-1] != target.shape[-1]:
                    raise ValueError('Unexpected reader signal')
                tangent = (target - target.detach()).to(device=signal.device, dtype=signal.dtype)
                return (signal + coefficient.to(signal.device, signal.dtype)[:, None] * tangent,)
            handles.append(native.read_out.register_forward_pre_hook(inject))
        yield
    finally:
        for handle in handles:
            handle.remove()
