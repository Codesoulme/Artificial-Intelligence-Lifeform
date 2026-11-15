"""Natural-language routing for the isolated semantic-slot prototype."""
from dataclasses import dataclass
import re

import torch

from .semantic_slot_state import SemanticSlotState, load as load_state, save as save_state


_FACT = re.compile(r'^\s*(?P<subject>[^。！？!?]{1,64}?)(?:最)?喜欢喝(?P<value>[^。！？!?]{1,64}?)\s*[。！？!?]?\s*$')
_QUESTION = re.compile(r'^\s*(?P<subject>[^。！？!?]{1,64}?)(?:最)?喜欢喝什么(?:饮料|饮品)\s*[？?。！!]?$')


@dataclass(frozen=True)
class ParsedFact:
    subject: str
    value: str


def parse_fact(text):
    if not isinstance(text, str):
        raise ValueError('Fact text must be a string')
    match = _FACT.fullmatch(text)
    if match is None:
        raise ValueError('Unsupported fact form')
    subject, value = match.group('subject').strip(), match.group('value').strip()
    if not subject or not value:
        raise ValueError('Fact subject and value must be nonempty')
    return ParsedFact(subject, value)


def parse_question(text):
    if not isinstance(text, str):
        raise ValueError('Question text must be a string')
    match = _QUESTION.fullmatch(text)
    if match is None:
        raise ValueError('Unsupported question form')
    subject = match.group('subject').strip()
    if not subject:
        raise ValueError('Question subject must be nonempty')
    return subject


class NaturalSemanticSlotMemory:
    """Route parsed facts into a versioned semantic slot bank."""

    def __init__(self, state, *, base_sha256, bridge_sha256):
        if not isinstance(state, SemanticSlotState):
            raise ValueError('A semantic slot state is required')
        self.state = state
        self.base_sha256 = base_sha256
        self.bridge_sha256 = bridge_sha256

    def write(self, text, tokenizer, embedding):
        fact = parse_fact(text)
        token_ids = tokenizer.encode(fact.value, add_special_tokens=False)
        if not token_ids:
            raise ValueError('Fact value has no tokenizer representation')
        if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
                or token_ids[0] >= embedding.shape[0]):
            raise ValueError('Embedding table cannot represent fact value')
        vector = embedding[token_ids[0]].detach().float().cpu()
        keys = list(getattr(self, '_keys', [None] * self.state.max_slots))
        if len(keys) != self.state.max_slots:
            raise ValueError('Semantic slot key table is inconsistent')
        replacing = fact.subject in keys
        try:
            index = keys.index(fact.subject)
        except ValueError:
            try:
                index = keys.index(None)
            except ValueError:
                raise RuntimeError('Semantic slot bank is full')
        self.state = self.state.put(index, vector)
        keys[index] = fact.subject
        self._keys = tuple(keys)
        return {'action': 'replace' if replacing else 'new',
                'subject': fact.subject, 'value': fact.value, 'slot': index,
                'revision': self.state.revision}

    def query_slot(self, text):
        subject = parse_question(text)
        keys = getattr(self, '_keys', (None,) * self.state.max_slots)
        if subject not in keys:
            return {'subject': subject, 'slot': None, 'value': None}
        index = keys.index(subject)
        return {'subject': subject, 'slot': index, 'value': self.state.get(index)}

    def save(self, path, *, metadata=None):
        keys = list(getattr(self, '_keys', (None,) * self.state.max_slots))
        if len(keys) != self.state.max_slots or any(key is not None and not isinstance(key, str) for key in keys):
            raise ValueError('Invalid semantic slot key table')
        combined = dict(metadata or {})
        combined['slot_keys'] = keys
        return save_state(path, self.state, base_sha256=self.base_sha256,
                          bridge_sha256=self.bridge_sha256, metadata=combined)

    @classmethod
    def load(cls, path, *, expected_base_sha256, expected_bridge_sha256, device='cpu'):
        state, metadata = load_state(path, expected_base_sha256=expected_base_sha256,
                                     expected_bridge_sha256=expected_bridge_sha256, device=device)
        keys = metadata.get('metadata', {}).get('slot_keys')
        if not isinstance(keys, list) or len(keys) != state.max_slots:
            raise ValueError('Semantic slot key table is missing')
        if any(key is not None and (not isinstance(key, str) or not key.strip()) for key in keys):
            raise ValueError('Semantic slot key table is invalid')
        result = cls(state, base_sha256=expected_base_sha256, bridge_sha256=expected_bridge_sha256)
        result._keys = tuple(keys)
        return result
