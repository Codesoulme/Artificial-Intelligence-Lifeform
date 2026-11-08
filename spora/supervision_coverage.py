"""Independent accounting for observed supervised document positions.

Coordinates must refer to the immutable file identified by its SHA-256 digest.
Only submit positions actually supervised after all sampling and masking. This
module does not observe training or infer coverage for previous runs.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class CoverageSummary:
    intervals: tuple[tuple[int, int], ...]
    total_positions: int
    unique_positions: int
    duplicate_positions: int


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _source(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("source must be a SHA-256 hex digest")
    return value.lower()


class SupervisionCoverage:
    """Store interval observations, including repeated observations."""

    def __init__(self) -> None:
        self._records: list[tuple[str, int, int, int]] = []

    def add(self, source: str, document: int, start: int, end: int) -> None:
        source = _source(source)
        document = _integer(document, "document")
        start = _integer(start, "start")
        end = _integer(end, "end")
        if end <= start:
            raise ValueError("end must be greater than start")
        self._records.append((source, document, start, end))

    def summaries(self) -> dict[tuple[str, int], CoverageSummary]:
        groups: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for source, document, start, end in self._records:
            groups.setdefault((source, document), []).append((start, end))
        result = {}
        for key, intervals in groups.items():
            merged: list[tuple[int, int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            total = sum(end - start for start, end in intervals)
            unique = sum(end - start for start, end in merged)
            result[key] = CoverageSummary(tuple(merged), total, unique, total - unique)
        return result

    def to_dict(self) -> dict:
        """Return a JSON-compatible, versioned snapshot of all observations."""
        return {"version": 1, "records": [
            {"source": source, "document": document, "start": start, "end": end}
            for source, document, start, end in self._records
        ]}

    @classmethod
    def from_dict(cls, state: object) -> "SupervisionCoverage":
        """Validate and restore a snapshot, rejecting unknown fields."""
        if (not isinstance(state, dict) or set(state) != {"version", "records"}
                or type(state["version"]) is not int or state["version"] != 1
                or not isinstance(state["records"], list)):
            raise ValueError("invalid coverage snapshot")
        result = cls()
        for record in state["records"]:
            if not isinstance(record, dict) or set(record) != {"source", "document", "start", "end"}:
                raise ValueError("invalid coverage record")
            result.add(record["source"], record["document"], record["start"], record["end"])
        return result
