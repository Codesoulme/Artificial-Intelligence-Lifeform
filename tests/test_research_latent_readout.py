import unittest

import torch

from spora.research_latent_readout import LatentReadout


class LatentReadoutTests(unittest.TestCase):
    def test_initial_identity_and_value_sensitive_incremental_reads(self):
        torch.manual_seed(12)
        reader = LatentReadout(8, rank=4)
        hidden = torch.randn(1, 5, 8)
        slots = torch.randn(1, 2, 8)
        initial = reader(hidden, slots)
        torch.testing.assert_close(initial, hidden, rtol=0, atol=0)
        initial.square().mean().backward()
        self.assertGreater(reader.out.weight.grad.norm().item(), 0.)
        with torch.no_grad():
            reader.out.weight.uniform_(-.1, .1)
            updated = reader(hidden, slots)
            torch.testing.assert_close(updated[:, -1:], reader(hidden[:, -1:], slots))
            replacement = slots.clone()
            replacement[:, 0] *= -1
            self.assertFalse(torch.equal(updated, reader(hidden, replacement)))
            with self.assertRaises(ValueError):
                reader(hidden, slots[:, :0])


if __name__ == '__main__':
    unittest.main()
