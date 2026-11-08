"""Fail-closed admission checks for a reviewed training corpus.

This policy is independent of the trainer and never changes data or approves a
source implicitly. Callers must bind the exact artifact digest themselves.
"""
from __future__ import annotations


def validate_review(review: dict, *, artifact_sha256: str, schedule_steps: int,
                    max_major_rate: float = 0.0, max_unsafe: int = 0,
                    require_independent_review: bool = True) -> dict:
    if not isinstance(review, dict) or review.get("approved_for_pilot") is not True:
        raise ValueError("corpus review is not approved for pilot")
    if type(schedule_steps) is not int or schedule_steps < 1:
        raise ValueError("invalid schedule")
    if type(review.get("max_schedule_steps")) is not int or review["max_schedule_steps"] < schedule_steps:
        raise ValueError("review does not cover schedule")
    if review.get("training_source_sha256") is None or review["training_source_sha256"].get("artifact") != artifact_sha256:
        raise ValueError("review does not bind exact artifact")
    quality = review.get("quality_gate")
    if not isinstance(quality, dict):
        raise ValueError("missing quality gate")
    if quality.get("status") != "passed":
        raise ValueError("quality gate has not passed")
    if require_independent_review and quality.get("independent_review") is not True:
        raise ValueError("independent review is required")
    sample_count = quality.get("sample_count")
    major = quality.get("major_count")
    unsafe = quality.get("unsafe_count")
    if (type(sample_count) is not int or sample_count < 1 or type(major) is not int or major < 0
            or type(unsafe) is not int or unsafe < 0 or major > sample_count or unsafe > sample_count):
        raise ValueError("invalid quality counts")
    if major / sample_count > max_major_rate or unsafe > max_unsafe:
        raise ValueError("quality gate exceeds configured limits")
    return {"artifact_sha256": artifact_sha256, "schedule_steps": schedule_steps,
            "sample_count": sample_count, "major_count": major, "unsafe_count": unsafe,
            "independent_review": quality["independent_review"]}
