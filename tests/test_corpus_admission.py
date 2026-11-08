import unittest

from spora.corpus_admission import validate_review


def review(**quality):
    return {"approved_for_pilot": True, "max_schedule_steps": 100,
            "training_source_sha256": {"artifact": "a"},
            "quality_gate": {"status": "passed", "independent_review": True,
                             "sample_count": 100, "major_count": 0, "unsafe_count": 0, **quality}}


class CorpusAdmissionTests(unittest.TestCase):
    def test_accepts_bound_clean_review(self):
        result = validate_review(review(), artifact_sha256="a", schedule_steps=100)
        self.assertEqual(result["major_count"], 0)

    def test_rejects_missing_quality_gate_and_major(self):
        bad = review(status="pending")
        with self.assertRaisesRegex(ValueError, "quality gate"):
            validate_review(bad, artifact_sha256="a", schedule_steps=100)
        bad = review(major_count=1)
        with self.assertRaisesRegex(ValueError, "quality gate exceeds"):
            validate_review(bad, artifact_sha256="a", schedule_steps=100)

    def test_rejects_wrong_artifact_or_independence(self):
        with self.assertRaisesRegex(ValueError, "exact artifact"):
            validate_review(review(), artifact_sha256="b", schedule_steps=100)
        with self.assertRaisesRegex(ValueError, "independent"):
            validate_review(review(independent_review=False), artifact_sha256="a", schedule_steps=100)


if __name__ == "__main__":
    unittest.main()
