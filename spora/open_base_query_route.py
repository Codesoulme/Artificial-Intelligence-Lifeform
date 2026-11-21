"""Small research router that protects open-base memory reads from false links.

The route is an address index, not the stored fact. It is intentionally kept
outside the language prompt: a missing address must veto the neural reader
instead of allowing a learned address head to select the nearest known slot.
"""
from dataclasses import dataclass
import re


_NUMERIC_OBJECT = re.compile(r'(?:对象|编号)\s*(\d+)')


@dataclass(frozen=True)
class RecordedAddressIndex:
    key_count: int
    keys: tuple

    format = 'spora-recorded-address-index-v1'

    def __post_init__(self):
        if type(self.key_count) is not int or self.key_count < 1:
            raise ValueError('key_count must be a positive integer')
        if (not isinstance(self.keys, tuple)
                or any(type(key) is not int or not 0 <= key < self.key_count for key in self.keys)
                or tuple(sorted(set(self.keys))) != self.keys):
            raise ValueError('keys must be a sorted tuple of valid unique addresses')

    @classmethod
    def empty(cls, key_count):
        return cls(key_count, ())

    @classmethod
    def from_keys(cls, key_count, keys):
        return cls(key_count, tuple(sorted(set(keys))))

    def contains(self, key):
        return type(key) is int and key in self.keys

    def add(self, key):
        if type(key) is not int or not 0 <= key < self.key_count:
            raise ValueError('Address is outside the configured key range')
        return RecordedAddressIndex.from_keys(self.key_count, (*self.keys, key))

    def payload(self):
        return {'format': self.format, 'key_count': self.key_count, 'keys': list(self.keys)}

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict) or payload.get('format') != cls.format:
            raise ValueError('Unexpected address index format')
        return cls.from_keys(payload['key_count'], payload['keys'])


def parse_numeric_object(question, key_count):
    if not isinstance(question, str) or not question:
        return None
    matches = _NUMERIC_OBJECT.findall(question)
    if len(matches) != 1:
        return None
    key = int(matches[0])
    return key if 0 <= key < key_count else None


def resolve_numeric_object(question, index):
    if not isinstance(index, RecordedAddressIndex):
        raise TypeError('A RecordedAddressIndex is required')
    key = parse_numeric_object(question, index.key_count)
    return key if key is not None and index.contains(key) else None
