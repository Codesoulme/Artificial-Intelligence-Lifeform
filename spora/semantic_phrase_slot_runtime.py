"""Natural-language routing for persisted semantic phrase slots."""
import hashlib

import torch

from .semantic_slot_runtime import parse_fact, parse_question_route
from .semantic_phrase_slot_state import SemanticPhraseSlotState, load as load_state, save as save_state


SEQUENCE_BRIDGE_BINDING = hashlib.sha256(b'semantic-sequence-scale-32-v1').hexdigest()


class NaturalSemanticPhraseMemory:
    def __init__(self, state, *, base_sha256, bridge_sha256=SEQUENCE_BRIDGE_BINDING):
        if not isinstance(state, SemanticPhraseSlotState):
            raise ValueError('A semantic phrase slot state is required')
        self.state = state
        self.base_sha256 = base_sha256
        self.bridge_sha256 = bridge_sha256

    def write(self, text, tokenizer, embedding):
        fact = parse_fact(text)
        token_ids = tokenizer.encode(fact.value, add_special_tokens=False)
        if not token_ids:
            raise ValueError('Fact value has no tokenizer representation')
        token_ids = token_ids + [tokenizer.eos_token_id]
        if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
                or any(token_id < 0 or token_id >= embedding.shape[0] for token_id in token_ids)):
            raise ValueError('Embedding table cannot represent fact value')
        sequence = embedding[torch.tensor(token_ids, device=embedding.device)].detach().float()
        return self.write_encoded_sequence(text, sequence, token_count=len(token_ids) - 1)

    def write_encoded_sequence(self, text, sequence, *, token_count=None):
        fact = parse_fact(text)
        if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 2
                or sequence.shape[1] != self.state.width or not sequence.shape[0]
                or not sequence.is_floating_point() or not bool(torch.isfinite(sequence).all())):
            raise ValueError('Encoded phrase sequence is invalid')
        keys = list(getattr(self, '_keys', [None] * self.state.max_slots))
        if len(keys) != self.state.max_slots:
            raise ValueError('Semantic phrase slot key table is inconsistent')
        key = f'{fact.subject}|{fact.relation}'
        replacing = key in keys
        try:
            index = keys.index(key)
        except ValueError:
            try:
                index = keys.index(None)
            except ValueError:
                raise RuntimeError('Semantic phrase slot bank is full')
        self.state = self.state.put(index, sequence)
        keys[index] = key
        self._keys = tuple(keys)
        return {'action': 'replace' if replacing else 'new', 'subject': fact.subject,
                'relation': fact.relation, 'value': fact.value, 'slot': index,
                'token_count': token_count if token_count is not None else sequence.shape[0],
                'revision': self.state.revision}

    def query_slot(self, text):
        subject, relation = parse_question_route(text)
        keys = getattr(self, '_keys', (None,) * self.state.max_slots)
        key = f'{subject}|{relation}'
        if key not in keys:
            return {'subject': subject, 'relation': relation, 'slot': None, 'value': None}
        index = keys.index(key)
        return {'subject': subject, 'relation': relation, 'slot': index,
                'value': self.state.get(index)}

    def save(self, path, *, metadata=None):
        keys = list(getattr(self, '_keys', (None,) * self.state.max_slots))
        if len(keys) != self.state.max_slots or any(key is not None and not isinstance(key, str) for key in keys):
            raise ValueError('Invalid semantic phrase slot key table')
        combined = dict(metadata or {})
        combined['slot_keys'] = keys
        return save_state(path, self.state, base_sha256=self.base_sha256,
                          bridge_sha256=self.bridge_sha256, metadata=combined)

    @classmethod
    def load(cls, path, *, expected_base_sha256, expected_bridge_sha256=SEQUENCE_BRIDGE_BINDING,
             device='cpu'):
        state, metadata = load_state(path, expected_base_sha256=expected_base_sha256,
                                     expected_bridge_sha256=expected_bridge_sha256, device=device)
        keys = metadata.get('metadata', {}).get('slot_keys')
        if not isinstance(keys, list) or len(keys) != state.max_slots:
            raise ValueError('Semantic phrase slot key table is missing')
        if any(key is not None and (not isinstance(key, str) or not key.strip()) for key in keys):
            raise ValueError('Semantic phrase slot key table is invalid')
        result = cls(state, base_sha256=expected_base_sha256,
                     bridge_sha256=expected_bridge_sha256)
        result._keys = tuple(keys)
        return result
