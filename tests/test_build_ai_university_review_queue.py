import unittest

from scripts.build_ai_university_review_queue import category, select


class AIUniversityQueueTests(unittest.TestCase):
    def test_balanced_categories_and_stability(self):
        rows = []
        for cat in ('a', 'b', 'c', 'd'):
            for i in range(20):
                rows.append({'record_id': f'{cat}-{i}', 'source_path': f'packages/content/{cat}/lessons/{i}.json'})
        queue = select(rows, 8)
        self.assertEqual(len(queue), 8)
        self.assertEqual({category(row) for row in queue}, {'a', 'b', 'c', 'd'})
        self.assertEqual(queue, select(rows, 8))


if __name__ == '__main__':
    unittest.main()
