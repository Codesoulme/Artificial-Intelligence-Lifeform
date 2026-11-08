"""Bind training-source approvals to the exact artifacts being consumed."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_training_source_reviews(root, plan):
    """Require an independently bound review for each enabled training source.

    Reviews may cover more than one source, but must explicitly state the hash
    of the exact artifact for each covered source. An approval for a small
    curriculum cannot implicitly approve a separate large dialogue corpus.
    """
    root = Path(root)
    data = root / plan["data"]
    base = root / plan.get("base_data", plan["data"])
    curriculum = root / plan["curriculum"]
    paths = {
        "pretrain": base / "pretrain-train.bin",
        "story": data / "pretrain-train.bin",
        "dialogue": data / "dialogue-train-tokens.jsonl",
        "teacher": curriculum / "teacher-tokens.jsonl",
        "foundation": curriculum / "foundations-tokens.jsonl",
    }
    if plan.get("focus_every", 0) <= plan["schedule_steps"]:
        paths["focus"] = data / "focus-tokens.json"
        active = set(plan["target_shares"]) | {"focus"}
    else:
        active = set(plan["target_shares"])
    reviews = plan.get("training_source_reviews", {})
    verified = {}
    for kind in sorted(active):
        if kind not in paths or kind not in reviews:
            raise ValueError("Missing source-specific review for " + kind)
        ref = reviews[kind]
        review_path = root / ref["path"]
        if file_sha256(review_path) != ref["sha256"]:
            raise ValueError("Source review changed: " + kind)
        review = json.loads(review_path.read_text(encoding="utf8"))
        if review.get("approved_for_pilot") is not True:
            raise ValueError("Source review not approved: " + kind)
        limit = review.get("max_schedule_steps")
        if type(limit) is not int or limit < plan["schedule_steps"]:
            raise ValueError("Source review does not cover schedule: " + kind)
        expected = review.get("training_source_sha256", {}).get(kind)
        actual = file_sha256(paths[kind])
        if not isinstance(expected, str) or expected.lower() != actual:
            raise ValueError("Source review covers different artifact: " + kind)
        verified[kind] = {"artifact_sha256": actual, "review_sha256": ref["sha256"]}
    return verified
