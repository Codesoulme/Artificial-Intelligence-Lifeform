import unittest

from scripts.prepare_crosswoz_unflagged_review import choose


class UnflaggedReviewTests(unittest.TestCase):
    def test_selection_is_deterministic_and_never_includes_flagged_rows(self):
        data = {str(i): {'type': 'fixture', 'messages': [{'role': 'usr', 'content': str(i)}]} for i in range(5)}
        flags = [{'dialogue_id': str(i), 'flags': ['phone'] if i == 0 else []} for i in range(5)]
        rows, population = choose(data, flags, 3)
        self.assertEqual(population, 4)
        self.assertEqual(rows, choose(dict(reversed(list(data.items()))), list(reversed(flags)), 3)[0])
        self.assertNotIn('0', {row['dialogue_id'] for row in rows})
        with self.assertRaisesRegex(ValueError, 'cover all'):
            choose(data, flags[:-1], 3)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            choose(data, flags + [flags[0]], 3)


if __name__ == '__main__':
    unittest.main()
