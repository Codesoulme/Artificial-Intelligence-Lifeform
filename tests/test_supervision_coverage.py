import json
import random
import unittest

from spora.supervision_coverage import SupervisionCoverage


class SupervisionCoverageTests(unittest.TestCase):
    def test_random_intervals_match_position_reference(self):
        rng = random.Random(123)
        coverage = SupervisionCoverage()
        expected = []
        for _ in range(300):
            start = rng.randrange(50)
            end = start + rng.randrange(1, 10)
            coverage.add('c' * 64, 4, start, end)
            expected.extend(range(start, end))
        result = coverage.summaries()[('c' * 64, 4)]
        self.assertEqual(result.total_positions, len(expected))
        self.assertEqual(result.unique_positions, len(set(expected)))
        self.assertEqual(result.duplicate_positions, len(expected) - len(set(expected)))

    def test_overlap_adjacent_and_repeated_observations(self):
        coverage = SupervisionCoverage()
        for start, end in [(0, 4), (2, 6), (6, 8), (2, 6), (10, 11)]:
            coverage.add("a" * 64, 0, start, end)
        summary = coverage.summaries()[("a" * 64, 0)]
        self.assertEqual(summary.intervals, ((0, 8), (10, 11)))
        self.assertEqual((summary.total_positions, summary.unique_positions, summary.duplicate_positions), (15, 9, 6))

    def test_sources_and_documents_are_independent(self):
        coverage = SupervisionCoverage()
        coverage.add("A" * 64, 0, 0, 3)
        coverage.add("b" * 64, 0, 0, 3)
        coverage.add("a" * 64, 1, 0, 3)
        groups = coverage.summaries()
        self.assertEqual(set(groups), {("a" * 64, 0), ("b" * 64, 0), ("a" * 64, 1)})
        self.assertTrue(all(s.unique_positions == 3 and s.duplicate_positions == 0 for s in groups.values()))

    def test_json_round_trip_preserves_observations_and_future_updates(self):
        coverage = SupervisionCoverage()
        coverage.add("a" * 64, 7, 3, 8)
        coverage.add("a" * 64, 7, 3, 8)
        restored = SupervisionCoverage.from_dict(json.loads(json.dumps(coverage.to_dict())))
        self.assertEqual(restored.to_dict(), coverage.to_dict())
        restored.add("a" * 64, 7, 7, 10)
        summary = restored.summaries()[("a" * 64, 7)]
        self.assertEqual((summary.unique_positions, summary.duplicate_positions), (7, 6))
        self.assertEqual(len(coverage.to_dict()["records"]), 2)

    def test_invalid_input_and_snapshots_are_rejected(self):
        coverage = SupervisionCoverage()
        for args in [("x" * 64, 0, 0, 1), ("a" * 63, 0, 0, 1),
                     ("a" * 64, -1, 0, 1), ("a" * 64, True, 0, 1),
                     ("a" * 64, 0, -1, 1), ("a" * 64, 0, 1, 1),
                     ("a" * 64, 0, 2, 1), ("a" * 64, 0, 0, 1.5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                coverage.add(*args)
        self.assertEqual(coverage.to_dict()["records"], [])
        for state in [None, {}, {"version": True, "records": []},
                      {"version": 2, "records": []}, {"version": 1, "records": {}},
                      {"version": 1, "records": [{"source": "a" * 64, "document": 0, "start": 0, "end": 1, "text": "secret"}]},
                      {"version": 1, "records": [{"source": "a" * 64, "document": 0, "start": 1, "end": 1}]}]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                SupervisionCoverage.from_dict(state)


if __name__ == "__main__":
    unittest.main()
