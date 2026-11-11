"""Checks for the research prefix reader's handling of BF16 weights."""
import io
import json
import struct
import unittest

import torch
from safetensors.torch import save

from scripts.deepseek_spora_prefix_features import read_bf16_tensor


class PrefixWeightReaderTest(unittest.TestCase):
    def test_matches_safetensors_values(self):
        expected = torch.tensor([[1.25, -2.5], [0.0, 99.0]], dtype=torch.bfloat16)
        payload = save({'model.test': expected})
        length = struct.unpack('<Q', payload[:8])[0]
        header = json.loads(payload[8:8 + length])
        actual = read_bf16_tensor(io.BytesIO(payload), header, 8 + length, len(payload), 'model.test')
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_invalid_range_and_dtype(self):
        for item in (
            {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 6]},
            {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]},
            {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]},
        ):
            with self.subTest(item=item), self.assertRaises(ValueError):
                read_bf16_tensor(io.BytesIO(b'\0\0'), {'x': item}, 0, 2, 'x')


if __name__ == '__main__':
    unittest.main()
