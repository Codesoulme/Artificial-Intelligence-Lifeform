"""Exact-vocabulary bridge for isolated research; not a learned memory reader."""
import torch
from torch.nn import functional as F


def oracle_embeddings(raw, codebook, embedding, min_norm=1e-5):
    """Map valid slots to normalized table vectors; keep empty slots zero."""
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[0] != 1
            or not isinstance(codebook, torch.Tensor) or codebook.ndim != 2
            or raw.shape[-1] != codebook.shape[-1]
            or not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
            or embedding.shape[0] != codebook.shape[0] or min_norm < 0):
        raise ValueError('Invalid oracle bridge tensors')
    valid = raw.norm(dim=-1) > min_norm
    flattened = F.normalize(raw.reshape(-1, raw.shape[-1]).float(), dim=-1)
    ids = (flattened @ codebook.float().T).argmax(dim=-1)
    exact = embedding.index_select(0, ids).float().reshape(1, raw.shape[1], -1)
    return F.normalize(exact, dim=-1) * valid.unsqueeze(-1)


def context_oracle_embeddings(decoder, codec, embedding, codes):
    """Read active research matrices and decode their slots using a shared table."""
    active = decoder._memory_context.get()
    if active is None:
        raise ValueError('Oracle bridge requires an explicit memory context')
    matrices, selected, mask, slots = active
    if (selected is None or len(matrices) != 2 or codes.ndim != 3 or codes.shape[0] != 1
            or codes.shape[1] > codec.slot_count):
        raise ValueError('Unsupported oracle bridge call')
    if mask is not None and not bool(mask[0]):
        raise ValueError('Unexpected masked bridge call')
    reads = []
    for matrix, address in zip(matrices, selected):
        key = F.normalize(address.float(), dim=-1)
        reads.append(torch.bmm(matrix[0].float(), key.unsqueeze(-1)).squeeze(-1))
    raw = ((reads[0] + reads[1]) * 0.5).reshape(1, codec.slot_count, codec.slot_width)[:, :codes.shape[1]]
    return oracle_embeddings(raw, codec.codebook, embedding)
