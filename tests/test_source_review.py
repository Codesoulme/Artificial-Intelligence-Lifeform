import json
import tempfile
import unittest
from pathlib import Path

from spora.source_review import file_sha256, verify_training_source_reviews


class SourceReviewTests(unittest.TestCase):
    def test_review_must_bind_each_active_artifact_and_reject_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "curriculum").mkdir()
            dialogue = root / "data/dialogue-train-tokens.jsonl"
            focus = root / "data/focus-tokens.json"
            dialogue.write_text("dialogue", encoding="utf8")
            focus.write_text("focus", encoding="utf8")
            review = root / "review.json"
            review.write_text(json.dumps({"approved_for_pilot": True, "max_schedule_steps": 100, "training_source_sha256": {
                "dialogue": file_sha256(dialogue), "focus": file_sha256(focus)}}), encoding="utf8")
            ref = {"path": "review.json", "sha256": file_sha256(review)}
            plan = {"data": "data", "curriculum": "curriculum", "schedule_steps": 100,
                    "focus_every": 10, "target_shares": {"dialogue": 1.0},
                    "training_source_reviews": {"dialogue": ref, "focus": ref}}
            self.assertEqual(set(verify_training_source_reviews(root, plan)), {"dialogue", "focus"})
            dialogue.write_text("changed", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "different artifact: dialogue"):
                verify_training_source_reviews(root, plan)

    def test_review_cannot_be_used_above_approved_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "curriculum").mkdir()
            teacher = root / "curriculum/teacher-tokens.jsonl"
            teacher.write_text("teacher", encoding="utf8")
            review = root / "review.json"
            review.write_text(json.dumps({"approved_for_pilot": True, "max_schedule_steps": 100,
                "max_target_shares": {"teacher": 0.01},
                "training_source_sha256": {"teacher": file_sha256(teacher)}}), encoding="utf8")
            plan = {"data": "data", "curriculum": "curriculum", "schedule_steps": 100,
                    "focus_every": 1000, "target_shares": {"teacher": 0.1},
                    "training_source_reviews": {"teacher": {"path": "review.json", "sha256": file_sha256(review)}}}
            with self.assertRaisesRegex(ValueError, "target share: teacher"):
                verify_training_source_reviews(root, plan)

    def test_curriculum_only_approval_does_not_approve_dialogue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "curriculum").mkdir()
            (root / "data/dialogue-train-tokens.jsonl").write_text("dialogue", encoding="utf8")
            review = root / "review.json"
            review.write_text(json.dumps({"approved_for_pilot": True, "max_schedule_steps": 100, "source_manifest_sha256": "some curriculum"}), encoding="utf8")
            plan = {"data": "data", "curriculum": "curriculum", "schedule_steps": 100,
                    "focus_every": 1000, "target_shares": {"dialogue": 1.0},
                    "training_source_reviews": {"dialogue": {"path": "review.json", "sha256": file_sha256(review)}}}
            with self.assertRaisesRegex(ValueError, "different artifact: dialogue"):
                verify_training_source_reviews(root, plan)


if __name__ == "__main__":
    unittest.main()
