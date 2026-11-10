import unittest
from unittest.mock import patch

import torch

from scripts.deepseek_read_mask_benchmark import legacy_forward
from spora.integrated_open_base import IntegratedOpenBase, _MemoryDecoderLayer
from tests.test_integrated_open_base import Base, make_reader, make_state


class ReadMaskBenchmarkTests(unittest.TestCase):
    def test_legacy_matches_selective_for_finite_values(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        inputs = torch.tensor([[[1., 2., 3.]], [[3., 2., 1.]]])
        for values in ([False, False], [True, False], [True, True]):
            mask = torch.tensor(values)
            with self.subTest(mask=values):
                selective = model(inputs, memory_state=make_state(1.), memory_read_mask=mask, current_only=True)
                with patch.object(_MemoryDecoderLayer, 'forward', legacy_forward):
                    legacy = model(inputs, memory_state=make_state(1.), memory_read_mask=mask, current_only=True)
                torch.testing.assert_close(selective, legacy)


if __name__ == '__main__':
    unittest.main()
