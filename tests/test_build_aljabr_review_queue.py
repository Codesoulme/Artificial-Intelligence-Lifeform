import unittest

from scripts.build_aljabr_review_queue import select_queue


class AlJabrQueueTests(unittest.TestCase):
    def test_queue_is_stable_and_covers_chapters(self):
        rows = []
        for source in ('one', 'two'):
            for chapter in range(3):
                for index in range(20):
                    rows.append({'record_id': f'{source}:{chapter}:{index}', 'source': source, 'path': f'chapter{chapter}.tex', 'block_index': index, 'chars': 100 + index, 'formula_placeholders': index % 10})
        queue = select_queue(rows, 20)
        self.assertEqual(len(queue), 20)
        self.assertEqual(queue, select_queue(rows, 20))
        self.assertEqual({(row['source'], row['path']) for row in queue}, {(s, f'chapter{c}.tex') for s in ('one', 'two') for c in range(3)})
        self.assertEqual({source: sum(row['source'] == source for row in queue) for source in ('one', 'two')}, {'one': 10, 'two': 10})


if __name__ == '__main__':
    unittest.main()
