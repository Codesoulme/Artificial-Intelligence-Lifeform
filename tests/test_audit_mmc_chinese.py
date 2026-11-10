import unittest

from scripts.audit_mmc_chinese import mentions, summarize


class MMCAuditTests(unittest.TestCase):
    def fixture(self):
        return {'tokens': ['\u6211', '\u559c\u6b22', '\u4f60', '\u6211'],
                'subtoken_map': [0, 0, 1, 2, 3, 3],
                'sentences': [['a', 'b', 'c'], ['d', 'e', 'f']],
                'speakers': [['A', 'A', 'A'], ['A', 'B', 'B']],
                'clusters': [[[0, 1], [4, 5]], [[2, 3]]]}

    def test_maps_subtokens_and_counts_distinct_speakers(self):
        document = self.fixture()
        result = list(mentions(document))
        self.assertEqual(result[0]['text'], '\u6211')
        self.assertEqual(result[1]['speaker'], 'B')
        self.assertEqual(result[2]['text'], '\u559c\u6b22\u4f60')
        summary = summarize([document])
        self.assertEqual(summary['first_singular'], 2)
        self.assertEqual(summary['multiple_first_person_different_speakers'], 1)

    def test_detects_cross_speaker_span(self):
        document = self.fixture()
        document['clusters'] = [[[3, 4]]]
        self.assertIsNone(list(mentions(document))[0]['speaker'])
        self.assertEqual(summarize([document])['multi_speaker_span'], 1)

    def test_rejects_bad_alignment_and_bounds(self):
        document = self.fixture()
        document['subtoken_map'][1] = 10
        with self.assertRaises(ValueError):
            list(mentions(document))
        document = self.fixture()
        document['clusters'] = [[[0, 6]]]
        with self.assertRaises(ValueError):
            list(mentions(document))


if __name__ == '__main__':
    unittest.main()
