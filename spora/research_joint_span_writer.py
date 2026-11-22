"""Research writer with a joint start/end span distribution."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import CHAR_BUCKETS, FIELDS, SPAN_LIMIT


class JointSpanWriter(nn.Module):
    """Score absent or complete character spans for every working-state field."""

    def __init__(self, width=64, *, feature_width=0, pair_width=16):
        super().__init__()
        if type(width) is not int or width <= 0 or type(pair_width) is not int or pair_width <= 0:
            raise ValueError('Writer widths must be positive integers')
        self.width = width
        self.pair_width = pair_width
        self.embedding = nn.Embedding(CHAR_BUCKETS, width, padding_idx=0)
        self.semantic = nn.Linear(feature_width, width, bias=False) if feature_width else None
        self.context = nn.ModuleList(
            nn.Conv1d(width, width, 5, dilation=d, padding=2 * d) for d in (1, 2, 4)
        )
        self.start = nn.Linear(width, len(FIELDS) * pair_width)
        self.end = nn.Linear(width, len(FIELDS) * pair_width)
        self.start_bias = nn.Linear(width, len(FIELDS))
        self.end_bias = nn.Linear(width, len(FIELDS))
        self.absent = nn.Linear(width, len(FIELDS))
        self.length_bias = nn.Parameter(torch.zeros(len(FIELDS), SPAN_LIMIT))

    def _features(self, codes, features):
        valid = codes.ne(0)
        x = self.embedding(codes)
        if self.semantic is not None:
            if features is None or features.shape[:2] != codes.shape:
                raise ValueError('Semantic features must align with character codes')
            x = x + self.semantic(F.layer_norm(features.float(), (features.shape[-1],)))
        x = (x * valid[:, :, None]).transpose(1, 2)
        for layer in self.context:
            x = (x + F.gelu(layer(x))) * valid[:, None, :]
        return x.transpose(1, 2), valid

    def forward(self, codes, features=None):
        x, valid = self._features(codes, features)
        batch, length, _ = x.shape
        start = self.start(x).reshape(batch, length, len(FIELDS), self.pair_width)
        end = self.end(x).reshape(batch, length, len(FIELDS), self.pair_width)
        pair = torch.einsum('btfk,bufk->bftu', start, end) / math.sqrt(self.pair_width)
        pair = pair + self.start_bias(x).permute(0, 2, 1).unsqueeze(-1)
        pair = pair + self.end_bias(x).permute(0, 2, 1).unsqueeze(-2)
        indices = torch.arange(length, device=codes.device)
        span_length = indices[None, :] - indices[:, None]
        valid_spans = (indices[None, :] >= indices[:, None]) & (span_length < SPAN_LIMIT)
        valid_spans = valid_spans & valid[:, :, None] & valid[:, None, :]
        length_index = span_length.clamp_min(0).clamp_max(SPAN_LIMIT - 1)
        pair = pair + self.length_bias[:, length_index].unsqueeze(0)
        pair = pair.masked_fill(~valid_spans[:, None, :, :], -torch.inf)
        absent = self.absent((x * valid[:, :, None]).sum(1) /
                             valid.sum(1, keepdim=True).clamp_min(1)).unsqueeze(-1)
        return torch.cat((absent, pair.flatten(2)), dim=-1)

    @torch.inference_mode()
    def write(self, text, margin=2., *, features=None):
        codes = torch.tensor(
            [[1 + ord(char) % (CHAR_BUCKETS - 1) for char in text]],
            dtype=torch.long, device=self.embedding.weight.device,
        )
        logits = self(codes, features)[0].cpu()
        length = len(text)
        spans = {}
        for field, scores in zip(FIELDS, logits, strict=True):
            best = int(scores[1:].argmax())
            start, end = divmod(best, length)
            if float(scores[best + 1] - scores[0]) >= margin:
                spans[field] = text[start:end + 1]
        return spans
