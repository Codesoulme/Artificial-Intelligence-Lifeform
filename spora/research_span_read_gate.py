"""A read decision that considers a prompt, a stored topic key, and available fields."""

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import FIELDS


class StateReadGate(nn.Module):
    def __init__(self, hidden, width=64):
        super().__init__()
        self.query = nn.Linear(hidden, width, bias=False)
        self.topic = nn.Linear(hidden, width, bias=False)
        self.need = nn.Linear(hidden, 1)
        self.fields = nn.Linear(hidden, len(FIELDS))
        self.match = nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(), nn.Linear(width, 1))

    def requirements(self, question):
        return self.need(question.float()).squeeze(-1), self.fields(question.float())

    def relevance(self, question, topic):
        question, topic = question.float(), topic.float()
        raw = F.cosine_similarity(question, topic, dim=-1)[..., None]
        q = F.normalize(self.query(question), dim=-1)
        k = F.normalize(self.topic(topic), dim=-1)
        return self.match(torch.cat((q * k, (q - k).abs(), raw), dim=-1)).squeeze(-1)

    @staticmethod
    def permits(need, required, match, present, thresholds):
        """Allow reads only if the intended state is relevant and contains the requested fields."""
        need_min, field_min, match_min = thresholds
        wanted = required >= field_min
        return ((need >= need_min) & (match >= match_min) & wanted.any(dim=-1)
                & ~(wanted & ~present).any(dim=-1))
