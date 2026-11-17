import unittest
import torch
from torch import nn

from spora.research_subject_task_heads import SubjectTask, SubjectTaskHeads


class SpyHead(nn.Module):
    def __init__(self, result):
        super().__init__()
        self.result = result
        self.calls = 0

    def forward(self, features, mask):
        self.calls += 1
        return torch.full((features.shape[0], 1), self.result)


class SubjectTaskHeadsTests(unittest.TestCase):
    def setUp(self):
        self.observation = SpyHead(1)
        self.query = SpyHead(2)
        self.heads = SubjectTaskHeads(self.observation, self.query)
        self.features = torch.zeros(2, 3, 4)
        self.mask = torch.ones(2, 3, dtype=torch.bool)

    def test_explicit_head_only(self):
        self.assertEqual(self.heads(self.features, self.mask, task=SubjectTask.OBSERVATION).tolist(), [[1], [1]])
        self.assertEqual((self.observation.calls, self.query.calls), (1, 0))
        self.assertEqual(self.heads(self.features, self.mask, task=SubjectTask.FACT_QUERY).tolist(), [[2], [2]])
        self.assertEqual((self.observation.calls, self.query.calls), (1, 1))

    def test_rejects_ambiguous_task_and_empty_mask_before_call(self):
        for task in (None, 'observation', 'fact_query'):
            with self.assertRaises(ValueError):
                self.heads(self.features, self.mask, task=task)
        with self.assertRaises(ValueError):
            self.heads(self.features, torch.zeros_like(self.mask), task=SubjectTask.OBSERVATION)
        self.assertEqual((self.observation.calls, self.query.calls), (0, 0))

    def test_disallows_sharing_same_head_instance(self):
        with self.assertRaises(ValueError):
            SubjectTaskHeads(self.observation, self.observation)
