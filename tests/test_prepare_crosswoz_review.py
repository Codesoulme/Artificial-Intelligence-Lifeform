import unittest

from scripts.prepare_crosswoz_review import make_queue


class CrossWOZReviewTests(unittest.TestCase):
    def test_deterministic_stratification_and_full_messages(self):
        data = {str(i): {'type': 'a' if i < 5 else 'b', 'messages': [
            {'role': 'usr', 'content': str(i)}, {'role': 'sys', 'content': '答复'}]} for i in range(10)}
        first = make_queue(data, per_type=2)
        self.assertEqual(first, make_queue(dict(reversed(list(data.items()))), per_type=2))
        self.assertEqual(len(first), 4)
        self.assertEqual([row['type'] for row in first], ['a', 'a', 'b', 'b'])
        self.assertTrue(all(len(row['messages']) == 2 for row in first))
        self.assertTrue(all(row['status'] == 'review_pending_not_approved' for row in first))


if __name__ == '__main__':
    unittest.main()
