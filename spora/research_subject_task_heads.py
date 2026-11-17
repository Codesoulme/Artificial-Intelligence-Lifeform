"""Research-only explicit dispatch between observation and fact-query heads.

This module does not infer task type, resolve identities, or authorize writes.
"""
from enum import Enum

import torch
from torch import nn


class SubjectTask(Enum):
    OBSERVATION = 'observation'
    FACT_QUERY = 'fact_query'


class SubjectTaskHeads(nn.Module):
    """Select exactly the caller-declared head, without fallbacks or score mixing."""

    def __init__(self, observation_head, fact_query_head):
        super().__init__()
        if not isinstance(observation_head, nn.Module) or not isinstance(fact_query_head, nn.Module):
            raise TypeError('Both heads must be modules')
        if observation_head is fact_query_head:
            raise ValueError('Task heads must be distinct module objects')
        self.observation_head = observation_head
        self.fact_query_head = fact_query_head

    def forward(self, features, mask, *, task):
        if not isinstance(task, SubjectTask):
            raise ValueError('Explicit SubjectTask required')
        if (features.ndim != 3 or mask.dtype != torch.bool or mask.shape != features.shape[:2]
                or not bool(mask.any(dim=1).all())):
            raise ValueError('Expected nonempty masked feature batch')
        head = self.observation_head if task is SubjectTask.OBSERVATION else self.fact_query_head
        return head(features, mask)
