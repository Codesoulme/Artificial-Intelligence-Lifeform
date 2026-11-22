"""Learned scoring of complete bounded character spans."""

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import CHAR_BUCKETS, FIELDS, SPAN_LIMIT, characters


def candidate_indices(length, device=None):
    starts, ends = [], []
    for start in range(length):
        for end in range(start, min(length, start + SPAN_LIMIT)):
            starts.append(start)
            ends.append(end)
    return (torch.tensor(starts, dtype=torch.long, device=device),
            torch.tensor(ends, dtype=torch.long, device=device))


class CandidateSpanWriter(nn.Module):
    """Choose one complete span per field from an explicit candidate set."""

    def __init__(self, width=64, *, feature_width=0):
        super().__init__()
        self.embedding = nn.Embedding(CHAR_BUCKETS, width, padding_idx=0)
        self.semantic = nn.Linear(feature_width, width, bias=False) if feature_width else None
        self.context = nn.ModuleList(
            nn.Conv1d(width, width, 5, dilation=d, padding=2 * d) for d in (1, 2, 4)
        )
        self.fields = nn.Embedding(len(FIELDS), width)
        self.lengths = nn.Embedding(SPAN_LIMIT, width)
        self.span = nn.Linear(width * 4, width)
        self.score = nn.Linear(width, 1)
        self.absent = nn.Linear(width, 1)

    def _features(self, codes, features):
        valid = codes.ne(0)
        x = self.embedding(codes)
        if self.semantic is not None:
            x = x + self.semantic(F.layer_norm(features.float(), (features.shape[-1],)))
        x = (x * valid[:, :, None]).transpose(1, 2)
        for layer in self.context:
            x = (x + F.gelu(layer(x))) * valid[:, None, :]
        return x.transpose(1, 2), valid

    def forward(self, codes, features=None):
        x, valid = self._features(codes, features)
        batch, length, width = x.shape
        starts, ends = candidate_indices(length, codes.device)
        span_lengths = ends - starts
        prefix = torch.cat((torch.zeros(batch, 1, width, device=x.device, dtype=x.dtype),
                            x.cumsum(dim=1)), dim=1)
        count = span_lengths.add(1).to(x.dtype).view(1, -1, 1)
        means = (prefix[:, ends + 1] - prefix[:, starts]) / count
        span = torch.cat((x[:, starts], x[:, ends], means,
                          self.lengths(span_lengths).unsqueeze(0).expand(batch, -1, -1)), dim=-1)
        span = F.gelu(self.span(span))
        field = self.fields.weight.view(1, len(FIELDS), 1, width)
        scored = self.score(torch.tanh(span[:, None] + field)).squeeze(-1)
        pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        absent = self.absent(torch.tanh(pooled[:, None] + self.fields.weight[None])).squeeze(-1)
        return torch.cat((absent.unsqueeze(-1), scored), dim=-1)

    @torch.inference_mode()
    def write(self, text, margin=2.0, *, features=None):
        codes = characters([text]).to(self.embedding.weight.device)
        logits = self(codes, features)[0].cpu()
        starts, ends = candidate_indices(len(text))
        spans = {}
        for field, scores in zip(FIELDS, logits, strict=True):
            best = int(scores[1:].argmax())
            if float(scores[best + 1] - scores[0]) >= margin:
                start, end = int(starts[best]), int(ends[best])
                spans[field] = text[start:end + 1]
        return spans
