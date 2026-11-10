import unittest

from scripts.audit_mmc_pronoun_pairs import pair_candidates
from scripts.deepseek_mmc_pair_probe import build_text, score, select_pairs


class MMCPairTests(unittest.TestCase):
    def document(self, key='scene_1'):
        return {'doc_key': key, 'tokens': ['\u6211', '\u559c\u6b22', '\u8336', '\u4f60', '\u5462'],
                'subtoken_map': [0, 1, 2, 3, 4],
                'speakers': [['Alice', 'Alice', 'Alice', 'Bob', 'Bob']],
                'clusters': [[[0, 0], [3, 3]]]}

    def test_extracts_pair_with_marker_and_anonymous_speakers(self):
        document = self.document()
        pair = list(pair_candidates(document))[0]
        self.assertTrue(pair['same_entity'])
        self.assertTrue(pair['heuristic_same'])
        text = build_text(document, pair)
        self.assertIn('<m1>\u6211</m1>', text)
        self.assertIn('<m2>\u4f60</m2>', text)
        self.assertNotIn('Alice', text)
        self.assertNotIn('Bob', text)
        self.assertNotIn('\u5462', text)

    def test_excludes_duplicate_span_across_clusters(self):
        document = self.document()
        document['clusters'].append([[3, 3]])
        self.assertEqual(list(pair_candidates(document)), [])

    def test_score_counts_false_acceptance_and_rejection(self):
        rows = [{'label': 0, 'speaker_count': 2}, {'label': 1, 'speaker_count': 3}]
        stats = score(rows, [1, 0])
        self.assertEqual(stats['all']['false_same'], 1)
        self.assertEqual(stats['all']['false_different'], 1)
        self.assertEqual(stats['two_speakers']['pairs'], 1)
        with self.assertRaises(ValueError):
            score(rows, [1])

    def test_selection_requires_distinct_scenes(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_pairs([self.document('same_1'), self.document('same_2')], 1)


if __name__ == '__main__':
    unittest.main()
