"""Continuous token codec for experimental neural text memory pages.

This module only defines the shared codebook contract. It does not decide
which entity an utterance names, persist a memory page, or provide a refusal
threshold for unknown addresses.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ContinuousTextCodec(nn.Module):
    """Pack short token sequences into fixed-width continuous memory slots."""

    def __init__(self, embedding, *, memory_width=None, slot_count=4, seed=29071, eos_token_id=None):
        super().__init__()
        if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
                or not embedding.is_floating_point() or min(embedding.shape) < 2):
            raise ValueError('A floating vocabulary embedding matrix is required')
        if type(slot_count) is not int or not 1 <= slot_count <= 32:
            raise ValueError('Slot count must be an integer in [1, 32]')
        target_width = embedding.shape[1] if memory_width is None else memory_width
        if type(target_width) is not int or target_width < 2 or target_width // slot_count < 2:
            raise ValueError('Embedding width cannot be divided into usable slots')
        if type(seed) is not int:
            raise ValueError('Seed must be an integer')
        if eos_token_id is not None and (type(eos_token_id) is not int or not 0 <= eos_token_id < embedding.shape[0]):
            raise ValueError('EOS token id is outside the vocabulary')
        self.vocab_size, self.embedding_width = embedding.shape
        self.memory_width = target_width
        self.slot_count = slot_count
        self.slot_width = target_width // slot_count
        self.eos_token_id = eos_token_id
        generator = torch.Generator(device='cpu').manual_seed(seed)
        projection = torch.randn((embedding.shape[1], self.slot_width), generator=generator,
                                 dtype=torch.float32, device='cpu').to(embedding.device)
        self.register_buffer('projection', projection)
        with torch.no_grad():
            codebook = F.normalize(embedding.float() @ projection, dim=-1)
        if not bool(torch.isfinite(codebook).all()):
            raise ValueError('Projected vocabulary contains non-finite values')
        self.register_buffer('codebook', codebook)

    @torch.no_grad()
    def encode(self, token_ids):
        """Return one fixed-width code for a token sequence plus optional EOS."""
        if (not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1
                or token_ids.dtype != torch.long or not token_ids.numel()
                or bool(((token_ids < 0) | (token_ids >= self.vocab_size)).any())):
            raise ValueError('Token ids must be a nonempty long vector in the vocabulary')
        if self.eos_token_id is None:
            sequence = token_ids
        else:
            if bool((token_ids == self.eos_token_id).any()):
                raise ValueError('Input must not contain the codec EOS token')
            sequence = torch.cat((token_ids, token_ids.new_tensor([self.eos_token_id])))
        if sequence.numel() > self.slot_count:
            raise ValueError('Token sequence exceeds codec slot capacity')
        result = torch.zeros((self.slot_count, self.slot_width), device=self.codebook.device,
                             dtype=self.codebook.dtype)
        codes = self.codebook.index_select(0, sequence.to(self.codebook.device))
        if bool((codes.norm(dim=-1) == 0).any()):
            raise ValueError('A selected token has no usable codebook vector')
        result[:sequence.numel()] = codes
        return result.reshape(-1)

    @torch.no_grad()
    def decode(self, code, *, min_slot_norm=0.0):
        """Decode slots, optionally stopping at a caller-calibrated minimum norm.

        The threshold only suppresses small numeric residuals. It cannot prove
        that a memory exists, identify an entity, or establish content accuracy.
        The default keeps the original exact-zero behavior.
        """
        if (not isinstance(code, torch.Tensor) or code.ndim != 1
                or code.numel() != self.slot_count * self.slot_width
                or not code.is_floating_point() or not bool(torch.isfinite(code).all())):
            raise ValueError('Code has the wrong finite floating shape')
        if (isinstance(min_slot_norm, bool) or not isinstance(min_slot_norm, (int, float))
                or not math.isfinite(min_slot_norm) or min_slot_norm < 0):
            raise ValueError('Minimum slot norm must be finite and nonnegative')
        slots = code.to(self.codebook.device).reshape(self.slot_count, self.slot_width)
        norms = slots.norm(dim=-1)
        tokens, margins = [], []
        for slot, vector in enumerate(slots):
            if float(norms[slot]) <= min_slot_norm:
                break
            scores = self.codebook @ F.normalize(vector, dim=0)
            top = scores.topk(2)
            token = int(top.indices[0])
            tokens.append(token)
            margins.append(float(top.values[0] - top.values[1]))
            if self.eos_token_id is not None and token == self.eos_token_id:
                break
        return {'token_ids': tokens, 'margins': margins, 'slot_norms': norms.tolist()}

    def config_dict(self):
        return {'vocab_size': self.vocab_size, 'embedding_width': self.embedding_width,
                'memory_width': self.memory_width,
                'slot_count': self.slot_count, 'slot_width': self.slot_width,
                'eos_token_id': self.eos_token_id}
