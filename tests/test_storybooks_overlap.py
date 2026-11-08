import tempfile
import unittest
from pathlib import Path

from scripts.storybooks_overlap import (blank, body, build, create_output_dir, finish,
                                        normalize, scan_record, write_results)


class DiagnosticTests(unittest.TestCase):
    def test_body_excludes_title_and_attribution(self):
        self.assertEqual(normalize(body('# title\n##\nfirst\n##\nsecond\n##\ncredit', 2)), 'firstsecond')

    def test_exact_fragment_ranges_and_repeated_occurrences(self):
        text = 'abcdefghijklmnopqrstuvwx'
        machine = build([text, text])
        results = [blank(), blank()]
        scan_record(machine, text + text, results, skip=1)
        result = finish(results)[0]
        self.assertEqual(result['documents_with_full_containment'], 1)
        self.assertEqual(result['documents_equal_full'], 0)
        self.assertEqual(result['windows']['20']['candidate_ranges'], [[0, 24]])
        self.assertEqual(result['windows']['20']['candidate_window_positions'], 5)
        self.assertEqual(result['windows']['20']['occurrences'], 10)
        self.assertEqual(results[1]['windows']['20']['documents'], 0)

    def test_output_uses_new_directories_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / 'runs'
            first = create_output_dir(runs)
            write_results(first, {'value': 1})
            with self.assertRaises(FileExistsError):
                write_results(first, {'value': 2})
            self.assertEqual((first / 'results.json').read_text(encoding='utf-8'),
                             '{\n  "value": 1\n}\n')
            second = create_output_dir(runs)
            self.assertNotEqual(first, second)
            self.assertEqual(second.parent, runs)
            self.assertFalse((second / 'results.json').exists())

    def test_equal_and_empty(self):
        machine = build(['abcdefghijkl'])
        result = [blank()]
        scan_record(machine, 'abcdefghijkl', result)
        self.assertEqual(result[0]['documents_equal_full'], 1)
        with self.assertRaises(ValueError):
            build([''])


if __name__ == '__main__':
    unittest.main()
