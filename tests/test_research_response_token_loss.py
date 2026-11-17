import unittest
import torch
from spora.research_response_token_loss import response_token_loss


class ResponseTokenLossTests(unittest.TestCase):
    def test_prompt_excluded_and_eos_scored(self):
        tokens=torch.tensor([[0,1,2,3]])
        logits=torch.zeros(1,4,4,requires_grad=True)
        loss=response_token_loss(logits,tokens,2)
        loss.backward()
        self.assertEqual(float(logits.grad[0,0].abs().sum()),0.0)
        self.assertGreater(float(logits.grad[0,1].abs().sum()),0.0)
        self.assertGreater(float(logits.grad[0,2].abs().sum()),0.0)
        self.assertEqual(float(logits.grad[0,3].abs().sum()),0.0)

    def test_invalid_prefix_rejected(self):
        with self.assertRaises(ValueError):
            response_token_loss(torch.zeros(1,3,4),torch.zeros(1,3,dtype=torch.long),3)
