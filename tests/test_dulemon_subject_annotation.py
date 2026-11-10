import unittest

from scripts.deepseek_dulemon_subject_annotation import compare
from scripts.deepseek_relative_identity_probe import ADDRESSEE, SPEAKER, UNKNOWN


class SubjectAnnotationTests(unittest.TestCase):
    def test_separates_wrong_subject_defer_and_unsafe_accept(self):
        rows = [{'sample_sha256': str(i), 'predicted': p} for i, p in enumerate(
            (SPEAKER, UNKNOWN, ADDRESSEE, SPEAKER))]
        labels = [(SPEAKER, 'a'), (ADDRESSEE, 'b'), (SPEAKER, 'c'), (UNKNOWN, 'd')]
        result = compare(rows, labels)
        self.assertEqual(result['single_correct'], 1)
        self.assertEqual(result['single_deferred'], 1)
        self.assertEqual(result['wrong_person'], 1)
        self.assertEqual(result['unsafe_accept'], 1)
        self.assertEqual(result['correct'], 1)

    def test_rejects_unmatched_annotations(self):
        with self.assertRaises(ValueError):
            compare([], [(SPEAKER, 'a')])


if __name__ == '__main__':
    unittest.main()
