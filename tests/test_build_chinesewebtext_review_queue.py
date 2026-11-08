import unittest

from scripts.build_chinesewebtext_review_queue import select_queue


class ChineseWebTextReviewQueueTests(unittest.TestCase):
    def test_selects_six_per_domain_then_fills_stably(self):
        rows = []
        for domain in ('a', 'b'):
            for index in range(8):
                rows.append({'record_id': f'{domain}-{index}', 'domain': {'single_label': domain}})
        queue = select_queue(rows, limit=10, per_domain=3)
        self.assertEqual(len(queue), 10)
        self.assertEqual(len({row['record_id'] for row in queue}), 10)
        self.assertEqual(select_queue(rows, limit=10, per_domain=3), queue)


if __name__ == '__main__':
    unittest.main()
