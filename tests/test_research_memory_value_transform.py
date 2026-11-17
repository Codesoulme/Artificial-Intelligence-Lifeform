import unittest
import torch
from spora.research_memory_value_transform import MemoryValueTransform


class MemoryValueTransformTests(unittest.TestCase):
    def test_identity_initialization_and_single_slot_gradient(self):
        transform=MemoryValueTransform(4)
        value=torch.randn(3,4)
        torch.testing.assert_close(transform(value),value)
        transform(value).square().sum().backward()
        self.assertGreater(float(transform.delta.weight.grad.norm()),0.0)
