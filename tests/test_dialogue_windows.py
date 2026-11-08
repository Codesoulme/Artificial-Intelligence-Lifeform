import unittest
from spora.dialogue_windows import dialogue_windows


class DialogueWindowsTests(unittest.TestCase):
    def test_overlap_keeps_context_and_supervises_each_target_once(self):
        tokens = list(range(19))
        mask = [False] + [i % 3 != 0 for i in range(1, len(tokens))]
        windows = list(dialogue_windows(tokens, mask, length=8, overlap=3))
        actual = []
        for start, values, valid in windows:
            self.assertLessEqual(len(values), 8)
            self.assertFalse(valid[0])
            actual.extend(start + i for i, flag in enumerate(valid) if flag)
        self.assertEqual(actual, [i for i in range(1, len(tokens)) if mask[i]])
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous[1][-3:], current[1][:3])

    def test_rejects_unaligned_or_invalid_windows(self):
        with self.assertRaises(ValueError):
            list(dialogue_windows([1, 2], [False]))
        with self.assertRaises(ValueError):
            list(dialogue_windows([1, 2], [False, True], length=8, overlap=8))
        with self.assertRaises(ValueError):
            list(dialogue_windows([1, 2], [0, 1]))


if __name__ == '__main__':
    unittest.main()
