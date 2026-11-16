import unittest
from scripts.deepseek_spora_interleaved_native_probe import schedule


class InterleavedNativeProbeTests(unittest.TestCase):
    def test_all_initial_facts_precede_all_corrections(self):
        events = schedule((4, 5, 6, 7))
        self.assertEqual([phase for _, _, phase in events], [0] * 4 + [1] * 4)
        self.assertEqual([index for index, _, _ in events[:4]], [index for index, _, _ in events[4:]])
        self.assertTrue(all(events[i][1] != events[i + 4][1] for i in range(4)))

    def test_rejects_duplicate_or_out_of_range_names(self):
        for indices in ((4, 4), (99,), ()):
            with self.assertRaises(ValueError):
                schedule(indices)


if __name__ == '__main__':
    unittest.main()
