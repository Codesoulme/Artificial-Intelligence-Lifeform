"""A typed boundary between language parsing and durable SPORA writes."""
from dataclasses import dataclass
import math


CLAIM_TYPES = ('affirmative', 'negation', 'hypothesis', 'third_party', 'action', 'question_history')
SOURCES = ('unknown', 'direct', 'report', 'inference', 'simulation')


@dataclass(frozen=True)
class ObservationCandidate:
    """Parsed evidence; it is not a memory write and has no side effects."""

    subject_key: int | None
    value_text: str | None
    value_index: int | None
    claim_type: str
    source: str = 'unknown'
    quality: float | None = None
    valid_at: int | None = None
    confidence: float = 0.0
    evidence_text: str = ''

    def validate(self, *, key_count=None, value_count=None):
        if self.claim_type not in CLAIM_TYPES:
            raise ValueError('Unknown claim type')
        if self.source not in SOURCES:
            raise ValueError('Unknown candidate source')
        if self.subject_key is not None and (type(self.subject_key) is not int or self.subject_key < 0):
            raise ValueError('Subject key must be a nonnegative integer or None')
        if key_count is not None and self.subject_key is not None and self.subject_key >= key_count:
            raise ValueError('Subject key is outside the individual address space')
        if self.value_index is not None and (type(self.value_index) is not int or self.value_index < 0):
            raise ValueError('Value index must be a nonnegative integer or None')
        if value_count is not None and self.value_index is not None and self.value_index >= value_count:
            raise ValueError('Value index is outside the finite memory vocabulary')
        if self.value_text is not None and (not isinstance(self.value_text, str) or not self.value_text.strip()):
            raise ValueError('Value text must be nonempty when supplied')
        for name, value in (('quality', self.quality), ('confidence', self.confidence)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{name} must be finite and in [0, 1]')
        if self.confidence is None:
            raise ValueError('confidence must be finite and in [0, 1]')
        if self.valid_at is not None and (type(self.valid_at) is not int or self.valid_at < 0):
            raise ValueError('valid_at must be a nonnegative integer or None')
        if not isinstance(self.evidence_text, str):
            raise ValueError('Evidence text must be a string')
        return self


def write_guard(candidate, *, key_count, value_count, current_cursor, minimum_confidence=0.9):
    """Return a decision for finite-memory persistence without performing it."""
    candidate.validate(key_count=key_count, value_count=value_count)
    if type(current_cursor) is not int or current_cursor < 0:
        raise ValueError('Current cursor must be a nonnegative integer')
    if (isinstance(minimum_confidence, bool) or not isinstance(minimum_confidence, (int, float))
            or not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1):
        raise ValueError('Minimum confidence must be finite and in [0, 1]')
    reasons = []
    if candidate.claim_type != 'affirmative':
        reasons.append('claim_is_not_affirmative')
    if candidate.subject_key is None:
        reasons.append('subject_unresolved')
    if candidate.value_index is None:
        reasons.append('finite_value_unresolved')
    if candidate.source not in ('direct', 'report'):
        reasons.append('source_untrusted_or_unknown')
    if candidate.quality is None:
        reasons.append('quality_unresolved')
    if candidate.valid_at is None or candidate.valid_at > current_cursor:
        reasons.append('effective_time_unresolved')
    if candidate.confidence < minimum_confidence:
        reasons.append('confidence_below_write_floor')
    return {'write': not reasons, 'reasons': reasons}
