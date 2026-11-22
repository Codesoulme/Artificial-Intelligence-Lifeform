"""A trigger-gated BIO span writer for compositional value copying."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from spora.research_dialogue_working_state import (
    CHAR_BUCKETS,
    FIELDS,
    SPAN_LIMIT,
    characters,
)


class TriggerBioValueWriter(nn.Module):
    """Predict field presence separately from a contiguous value span."""

    def __init__(self, width=64, *, feature_width=0):
        super().__init__()
        self.embedding = nn.Embedding(CHAR_BUCKETS, width, padding_idx=0)
        self.semantic = nn.Linear(feature_width, width, bias=False) if feature_width else None
        self.context = nn.ModuleList(
            nn.Conv1d(width, width, 5, dilation=d, padding=2 * d) for d in (1, 2, 4)
        )
        self.tags = nn.Linear(width, len(FIELDS) * 3)
        self.trigger = nn.Linear(width, len(FIELDS) * 2)

    def forward(self, codes, features=None):
        valid = codes.ne(0)
        x = self.embedding(codes)
        if self.semantic is not None:
            x = x + self.semantic(F.layer_norm(features.float(), (features.shape[-1],)))
        x = (x * valid[:, :, None]).transpose(1, 2)
        for layer in self.context:
            x = (x + F.gelu(layer(x))) * valid[:, None, :]
        x = x.transpose(1, 2)
        pooled = x.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        tags = self.tags(x).reshape(codes.shape[0], codes.shape[1], len(FIELDS), 3)
        tags = tags.permute(0, 2, 3, 1)
        triggers = self.trigger(pooled).reshape(codes.shape[0], len(FIELDS), 2)
        return tags, triggers

    @torch.inference_mode()
    def write(self, text, margin=2.0, *, features=None):
        codes = characters([text]).to(self.embedding.weight.device)
        tag_logits, trigger_logits = self(codes, features)
        tag_log_probs = tag_logits[0].log_softmax(dim=1).cpu()
        trigger = trigger_logits[0].cpu()
        length = len(text)
        spans = {}
        for field_index, field in enumerate(FIELDS):
            trigger_delta = float(trigger[field_index, 1] - trigger[field_index, 0])
            best_score = float('-inf')
            best = None
            for start in range(length):
                score = float(tag_log_probs[field_index, 1, start])
                for end in range(start, min(length, start + SPAN_LIMIT)):
                    if end > start:
                        score += float(tag_log_probs[field_index, 2, end])
                    score -= float(tag_log_probs[field_index, 0, start:end + 1].sum())
                    candidate = trigger_delta + score
                    if candidate > best_score:
                        best_score, best = candidate, (start, end)
            if best is not None and best_score >= margin:
                start, end = best
                spans[field] = text[start:end + 1]
        return spans
