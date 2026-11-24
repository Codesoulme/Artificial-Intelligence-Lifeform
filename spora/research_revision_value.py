"""Select one bounded value span or leave the record unchanged."""

import torch
from torch import nn
from torch.nn import functional as F


class RevisionValueHead(nn.Module):
    def __init__(self, hidden, field_count, span_limit=12, *, contextual=False):
        super().__init__()
        self.span_limit = span_limit
        self.project = nn.Linear(hidden, 64)
        self.field = nn.Embedding(field_count, 64)
        self.context = nn.GRU(64, 32, batch_first=True, bidirectional=True) if contextual else None
        self.edges = nn.Linear(64, 2)
        self.ends = nn.Linear(64, 16, bias=False)
        self.abstain = nn.Linear(64, 1)

    def forward(self, hidden, valid, field):
        encoded = self.project(F.normalize(hidden.float(), dim=-1)) + self.field(field).unsqueeze(1)
        encoded = F.gelu(encoded)
        if self.context is not None:
            # Packing prevents the backward pass from seeing any right padding.
            lengths = valid.sum(dim=-1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
            context, _ = self.context(packed)
            contextual, _ = nn.utils.rnn.pad_packed_sequence(context, batch_first=True, total_length=encoded.shape[1])
            encoded = encoded + contextual
        start, end = self.edges(encoded).unbind(dim=-1)
        begin_key, end_key = self.ends(encoded).chunk(2, dim=-1)
        score = start.unsqueeze(-1) + end.unsqueeze(-2) + (begin_key @ end_key.transpose(-2, -1)) / 8 ** .5
        positions = torch.arange(valid.shape[-1], device=valid.device)
        offsets = positions[None, :] - positions[:, None]
        bounds = valid[:, :, None] & valid[:, None, :] & (offsets >= 0) & (offsets < self.span_limit)
        score = score.masked_fill(~bounds, -torch.inf)
        mean = encoded.masked_fill(~valid[..., None], 0.).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return torch.cat((self.abstain(mean), score.flatten(1)), dim=-1)


def select_value_spans(logits, sequence_width, *, floor=0.):
    selected, indices = logits[:, 1:].max(-1)
    accepted = (selected - logits[:, 0]) > floor
    return [(index // sequence_width, index % sequence_width + 1) if allowed else None
            for index, allowed in zip(indices.tolist(), accepted.tolist(), strict=True)]
