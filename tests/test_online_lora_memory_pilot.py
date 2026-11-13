import unittest

import torch
from torch import nn

from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta, observation_example


class ToyTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(len(text))),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


class OnlineLoraPilotTests(unittest.TestCase):
    def test_zero_initialization_and_disabled_path(self):
        torch.manual_seed(4)
        base = nn.Linear(3, 2)
        module = OnlineDelta(base, 2)
        inputs = torch.ones(1, 3)
        expected = base(inputs)
        self.assertTrue(torch.equal(module(inputs), expected))
        with torch.no_grad():
            module.up.weight.fill_(1.)
        self.assertFalse(torch.equal(module(inputs), expected))
        module.enabled = False
        self.assertTrue(torch.equal(module(inputs), expected))

    def test_observation_targets_only_the_known_value_span(self):
        fact, prefix, targets = observation_example(ToyTokenizer(), '甲', '果汁')
        self.assertEqual(fact, '甲喜欢喝果汁。')
        self.assertEqual(prefix, [0, 1, 2, 3])
        self.assertEqual(targets, [4, 5])


if __name__ == '__main__':
    unittest.main()
