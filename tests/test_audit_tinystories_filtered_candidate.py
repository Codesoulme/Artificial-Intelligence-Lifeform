import tempfile
import unittest
from pathlib import Path

from scripts.audit_tinystories_filtered_candidate import (
    candidate_internal,
    diagnostic_split,
    normalize,
    prior_overlap,
)


class TinyCandidateAuditTests(unittest.TestCase):
    def test_normalization_and_internal_duplicates(self):
        rows = [
            {'record_id': 'a', 'source_row_index_0based': 1, 'text': '甲 乙。'},
            {'record_id': 'b', 'source_row_index_0based': 2, 'text': '甲乙。'},
        ]
        self.assertEqual(normalize(rows[0]['text']), normalize(rows[1]['text']))
        result = candidate_internal(rows)
        self.assertEqual(result['exact_text']['duplicate_occurrences_beyond_first'], 1)
        self.assertTrue(result['source_row_indices_unique'])

    def test_partition_is_stable(self):
        self.assertEqual(diagnostic_split('story-1'), diagnostic_split('story-1'))
        self.assertIn(diagnostic_split('story-1')[0], {'train', 'development', 'test'})

    def test_prior_overlap_detects_full_and_leading_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.jsonl'
            text = '这是一个足够长的故事，包含候选故事的开头以及更多内容。' * 6
            path.write_text('{"text":' + repr(text).replace("'", '"') + '}\n', encoding='utf8')
            rows = [{'record_id': 'candidate', 'text': text}]
            result = prior_overlap(rows, {'old': path})
            self.assertEqual(result['records_with_exact_full_overlap'], 1)
            self.assertEqual(result['records_with_leading_96_containment'], 1)


if __name__ == '__main__':
    unittest.main()
