import unittest

from scripts.spora_temporal_address_probe import OUTPUT


class TemporalAddressProbeTests(unittest.TestCase):
    def test_probe_artifact_is_scoped_under_runs(self):
        self.assertTrue(OUTPUT.is_absolute())
        self.assertEqual(OUTPUT.parent.name, 'runs')
        self.assertIn('temporal-address', OUTPUT.name)


if __name__ == '__main__':
    unittest.main()
