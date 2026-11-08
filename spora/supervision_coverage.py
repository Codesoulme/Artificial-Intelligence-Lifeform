"""Independent accounting for observed supervised document positions.

Coordinates must refer to the immutable file identified by its SHA-256 digest.
Only submit positions actually supervised after all sampling and masking. This
module does not observe training or infer coverage for previous runs.
"""

from dataclasses import dataclass
from bisect import bisect_left
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
    """Store merged intervals and the total number of observed positions."""

    def __init__(self) -> None:
        self._groups: dict[tuple[str, int], tuple[list[tuple[int, int]], int]] = {}

    def add(self, source: str, document: int, start: int, end: int) -> None:
        source = _source(source)
        document = _integer(document, "document")
        start = _integer(start, "start")
        end = _integer(end, "end")
        if end <= start:
            raise ValueError("end must be greater than start")
        key = (source, document)
        intervals, total = self._groups.get(key, ([], 0))
        total += end - start
        index = bisect_left(intervals, (start,))
        if index and intervals[index - 1][1] >= start:
            index -= 1
        stop = index
        while stop < len(intervals) and intervals[stop][0] <= end:
            start = min(start, intervals[stop][0])
            end = max(end, intervals[stop][1])
            stop += 1
        intervals[index:stop] = [(start, end)]
        self._groups[key] = (intervals, total)

    def summaries(self) -> dict[tuple[str, int], CoverageSummary]:
        result = {}
        for key, (intervals, total) in self._groups.items():
            unique = sum(end - start for start, end in intervals)
            result[key] = CoverageSummary(tuple(intervals), total, unique, total - unique)
        return result

    def to_dict(self) -> dict:
        """Return a JSON-compatible snapshot of the accumulated coverage."""
        return {"version": 2, "records": [
            {"source": source, "document": document, "intervals": [list(pair) for pair in intervals],
             "total_positions": total}
            for (source, document), (intervals, total) in self._groups.items()
        ]}

    @classmethod
    def from_dict(cls, state: object) -> "SupervisionCoverage":
        """Validate and restore a snapshot, rejecting unknown fields."""
        if (not isinstance(state, dict) or set(state) != {"version", "records"}
                or type(state["version"]) is not int or state["version"] not in (1, 2)
                or not isinstance(state["records"], list)):
            raise ValueError("invalid coverage snapshot")
        result = cls()
        for record in state["records"]:
            if state["version"] == 1:
                if not isinstance(record, dict) or set(record) != {"source", "document", "start", "end"}:
                    raise ValueError("invalid coverage record")
                result.add(record["source"], record["document"], record["start"], record["end"])
                continue
            if not isinstance(record, dict) or set(record) != {"source", "document", "intervals", "total_positions"}:
                raise ValueError("invalid coverage record")
            key = (_source(record["source"]), _integer(record["document"], "document"))
            if key in result._groups or not isinstance(record["intervals"], list) or not record["intervals"]:
                raise ValueError("invalid coverage intervals or duplicate group")
            intervals = []
            unique = 0
            for pair in record["intervals"]:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("invalid coverage interval")
                start = _integer(pair[0], "start")
                end = _integer(pair[1], "end")
                if end <= start or (intervals and start <= intervals[-1][1]):
                    raise ValueError("coverage intervals must be ordered and disjoint")
                intervals.append((start, end))
                unique += end - start
            total = _integer(record["total_positions"], "total_positions")
            if total < unique:
                raise ValueError("total positions cannot be less than unique positions")
            result._groups[key] = (intervals, total)
        return result
