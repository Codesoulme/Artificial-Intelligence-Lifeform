"""Structured identity/fact lanes for experimental continuous text memory.

The lane lengths are explicit metadata owned by the memory record. They are
not inferred from a question or silently recovered from a language model.
This module is a serialization primitive; it does not route natural language
or decide whether a fact is true.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StructuredTextRecord:
    packet: torch.Tensor
    metadata: dict
    revision: int


class StructuredTextMemoryState:
    """Persisted structured records accompanying a GrowingState snapshot."""

    FORMAT = 'spora-structured-text-state-v1'
    RELATIONAL_FORMAT = 'spora-structured-text-state-v2'

    def __init__(self, codec_config, records=None, revision=0, relations=None):
        self.codec_config = dict(codec_config)
        self.records = {} if records is None else dict(records)
        self.revision = revision
        self.relations = {} if relations is None else dict(relations)

    @staticmethod
    def tensor_name(key):
        return f'structured_text/{key}'

    def validate(self, memory_width, key_count=None):
        if not isinstance(self.codec_config, dict) or self.codec_config.get('memory_width') != memory_width:
            raise ValueError('Structured text codec width does not match the individual')
        identity_slots = self.codec_config.get('identity_slots')
        fact_slots = self.codec_config.get('fact_slots')
        if (type(identity_slots) is not int or type(fact_slots) is not int
                or identity_slots < 1 or fact_slots < 1):
            raise ValueError('Invalid structured text lane configuration')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('Invalid structured text revision')
        if not isinstance(self.records, dict):
            raise ValueError('Structured text records must be a mapping')
        if not isinstance(self.relations, dict):
            raise ValueError('Structured text relations must be a mapping')
        slots = set()
        for pair, slot in self.relations.items():
            if (not isinstance(pair, tuple) or len(pair) != 2
                    or type(pair[0]) is not int or pair[0] < 0
                    or (key_count is not None and pair[0] >= key_count)
                    or not isinstance(pair[1], str) or not pair[1].strip()
                    or pair[1] != pair[1].strip() or len(pair[1]) > 128
                    or type(slot) is not int or slot not in self.records or slot in slots):
                raise ValueError('Invalid structured text relation binding')
            slots.add(slot)
        for key, record in self.records.items():
            if type(key) is not int or key < 0 or (key_count is not None and key >= key_count):
                raise ValueError('Invalid structured text record key')
            if (not isinstance(record, StructuredTextRecord)
                    or type(record.revision) is not int or record.revision < 1
                    or not isinstance(record.metadata, dict)
                    or set(record.metadata) != {'identity_length', 'fact_length'}
                    or type(record.metadata['identity_length']) is not int
                    or type(record.metadata['fact_length']) is not int
                    or not 1 <= record.metadata['identity_length'] <= identity_slots
                    or not 1 <= record.metadata['fact_length'] <= fact_slots):
                raise ValueError('Invalid structured text record metadata')
            if (not isinstance(record.packet, torch.Tensor) or record.packet.shape != (memory_width,)
                    or record.packet.dtype != torch.float32 or not bool(torch.isfinite(record.packet).all())):
                raise ValueError('Invalid structured text record packet')
            if record.revision > self.revision:
                raise ValueError('Structured text record revision is ahead of state')

    def put(self, key, packet, metadata, memory_width, key_count=None):
        if key in self.relations.values():
            raise ValueError('Relation-bound records require a relation write')
        return self._put(key, packet, metadata, memory_width, key_count, self.relations)

    def _put(self, key, packet, metadata, memory_width, key_count, relations):
        if type(key) is not int or key < 0 or (key_count is not None and key >= key_count):
            raise ValueError('Invalid structured text record key')
        if not isinstance(packet, torch.Tensor) or packet.shape != (memory_width,):
            raise ValueError('Structured text packet has the wrong shape')
        records = {index: StructuredTextRecord(value.packet.detach().clone(),
                                                dict(value.metadata), value.revision)
                   for index, value in self.records.items()}
        next_revision = self.revision + 1
        records[key] = StructuredTextRecord(packet.detach().clone().float(), dict(metadata), next_revision)
        result = StructuredTextMemoryState(self.codec_config, records, next_revision, relations)
        result.validate(memory_width, key_count)
        return result

    def put_relation(self, entity, relation, packet, metadata, memory_width, key_count, *, replace_existing=False):
        """Write an explicitly addressed relation; existing unbound records are untouched.

        Entity identity and relation semantics are caller supplied, not inferred.
        Corrections replace the current packet; no historical version is retained.
        """
        self.validate(memory_width, key_count)
        if (type(entity) is not int or type(key_count) is not int or key_count < 1
                or not 0 <= entity < key_count or not isinstance(relation, str)
                or not relation.strip() or relation != relation.strip() or len(relation) > 128
                or type(replace_existing) is not bool):
            raise ValueError('Invalid entity, relation, or replacement intent')
        pair = (entity, relation)
        relations = dict(self.relations)
        if pair in relations:
            if not replace_existing:
                raise ValueError('Relation already exists; explicit replacement required')
            slot = relations[pair]
        else:
            slot = next((i for i in range(key_count) if i not in self.records), None)
            if slot is None:
                raise ValueError('Structured text record capacity exhausted')
            relations[pair] = slot
        return self._put(slot, packet, metadata, memory_width, key_count, relations)

    def schema(self):
        self.validate(self.codec_config['memory_width'])
        schema = {
            'format': self.RELATIONAL_FORMAT if self.relations else self.FORMAT,
            'codec': dict(self.codec_config),
            'revision': self.revision,
            'records': [
                {'key': key, 'metadata': dict(record.metadata), 'revision': record.revision}
                for key, record in sorted(self.records.items())
            ],
        }
        if self.relations:
            schema['relations'] = [
                {'entity': entity, 'relation': relation, 'slot': slot}
                for (entity, relation), slot in sorted(self.relations.items())
            ]
        return schema

    @classmethod
    def from_schema(cls, schema, tensors, device='cpu'):
        if (not isinstance(schema, dict) or schema.get('format') not in (cls.FORMAT, cls.RELATIONAL_FORMAT)
                or not isinstance(schema.get('codec'), dict)
                or not isinstance(schema.get('records'), list)):
            raise ValueError('Invalid structured text state schema')
        records = {}
        for row in schema['records']:
            if (not isinstance(row, dict) or set(row) != {'key', 'metadata', 'revision'}
                    or type(row['key']) is not int or row['key'] in records):
                raise ValueError('Invalid structured text record schema')
            name = cls.tensor_name(row['key'])
            if name not in tensors:
                raise ValueError('Missing structured text record tensor')
            records[row['key']] = StructuredTextRecord(
                tensors[name].to(device), dict(row['metadata']), row['revision'])
        relations = {}
        if schema['format'] == cls.RELATIONAL_FORMAT:
            if not isinstance(schema.get('relations'), list) or not schema['relations']:
                raise ValueError('Invalid structured text relations schema')
            for row in schema['relations']:
                if (not isinstance(row, dict) or set(row) != {'entity', 'relation', 'slot'}
                        or type(row['entity']) is not int or not isinstance(row['relation'], str)):
                    raise ValueError('Invalid structured text relation schema')
                pair = (row['entity'], row['relation'])
                if pair in relations:
                    raise ValueError('Duplicate structured text relation')
                relations[pair] = row['slot']
        elif 'relations' in schema:
            raise ValueError('Unexpected relations in legacy schema')
        result = cls(schema['codec'], records, schema.get('revision'), relations)
        result.validate(schema['codec'].get('memory_width'))
        return result


class StructuredTextMemoryCodec(nn.Module):
    """Pack identity and fact token sequences into separate fixed slot lanes."""

    def __init__(self, embedding, *, memory_width=None, identity_slots=3,
                 fact_slots=5, identity_slot_width=None, fact_slot_width=None,
                 seed=29071, eos_token_id=None):
        super().__init__()
        if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
                or not embedding.is_floating_point() or min(embedding.shape) < 2):
            raise ValueError('A floating vocabulary embedding matrix is required')
        if (type(identity_slots) is not int or type(fact_slots) is not int
                or identity_slots < 1 or fact_slots < 1):
            raise ValueError('Lane slot counts must be positive integers')
        target_width = embedding.shape[1] if memory_width is None else memory_width
        if type(target_width) is not int or target_width < 2:
            raise ValueError('Memory width must be a positive integer')
        if identity_slot_width is None and fact_slot_width is None:
            if target_width % (identity_slots + fact_slots) != 0:
                raise ValueError('Memory width must divide into usable lane slots')
            identity_slot_width = fact_slot_width = target_width // (identity_slots + fact_slots)
        elif (type(identity_slot_width) is not int or type(fact_slot_width) is not int
              or identity_slot_width < 2 or fact_slot_width < 2
              or identity_slots * identity_slot_width + fact_slots * fact_slot_width != target_width):
            raise ValueError('Lane widths must exactly cover memory width')
        if type(seed) is not int:
            raise ValueError('Seed must be an integer')
        if (eos_token_id is not None
                and (type(eos_token_id) is not int or not 0 <= eos_token_id < embedding.shape[0])):
            raise ValueError('EOS token id is outside the vocabulary')
        self.vocab_size, self.embedding_width = embedding.shape
        self.memory_width = target_width
        self.identity_slots = identity_slots
        self.fact_slots = fact_slots
        self.total_slots = identity_slots + fact_slots
        self.identity_slot_width = identity_slot_width
        self.fact_slot_width = fact_slot_width
        self.slot_width = identity_slot_width if identity_slot_width == fact_slot_width else None
        self.seed = seed
        self.eos_token_id = eos_token_id
        generator = torch.Generator(device='cpu').manual_seed(seed)
        identity_projection = torch.randn((embedding.shape[1], identity_slot_width),
                                          generator=generator, dtype=torch.float32,
                                          device='cpu').to(embedding.device)
        fact_projection = torch.randn((embedding.shape[1], fact_slot_width),
                                      generator=generator, dtype=torch.float32,
                                      device='cpu').to(embedding.device)
        self.register_buffer('identity_projection', identity_projection)
        self.register_buffer('fact_projection', fact_projection)
        with torch.no_grad():
            identity_codebook = F.normalize(embedding.float() @ identity_projection, dim=-1)
            fact_codebook = F.normalize(embedding.float() @ fact_projection, dim=-1)
        if not bool(torch.isfinite(identity_codebook).all() and torch.isfinite(fact_codebook).all()):
            raise ValueError('Projected vocabulary contains non-finite values')
        self.register_buffer('identity_codebook', identity_codebook)
        self.register_buffer('fact_codebook', fact_codebook)

    def _ids(self, ids, lane, *, allow_empty=False):
        if isinstance(ids, torch.Tensor):
            if ids.ndim != 1 or ids.dtype != torch.long:
                raise ValueError(f'{lane} token ids must be a one-dimensional long tensor')
            values = ids.detach().cpu().tolist()
        elif isinstance(ids, (list, tuple)):
            values = list(ids)
        else:
            raise ValueError(f'{lane} token ids must be a sequence')
        if not values and not allow_empty:
            raise ValueError(f'{lane} token ids must not be empty')
        if any(type(value) is not int or value < 0 or value >= self.vocab_size for value in values):
            raise ValueError(f'{lane} token id is outside the vocabulary')
        if self.eos_token_id is not None and self.eos_token_id in values:
            raise ValueError(f'{lane} input must not contain the EOS token')
        return values

    def _pack_lane(self, ids, slots, slot_width, codebook, lane):
        values = self._ids(ids, lane)
        sequence = values if self.eos_token_id is None else values + [self.eos_token_id]
        if len(sequence) > slots:
            raise ValueError(f'{lane} token sequence exceeds its lane capacity')
        lane_code = codebook.new_zeros((slots, slot_width))
        lane_code[:len(sequence)] = codebook.index_select(
            0, torch.tensor(sequence, dtype=torch.long, device=codebook.device))
        return lane_code.reshape(-1), len(values)

    def encode(self, identity_ids, fact_ids):
        """Return a full memory vector and its boundary metadata."""
        identity, identity_length = self._pack_lane(
            identity_ids, self.identity_slots, self.identity_slot_width,
            self.identity_codebook, 'Identity')
        fact, fact_length = self._pack_lane(
            fact_ids, self.fact_slots, self.fact_slot_width,
            self.fact_codebook, 'Fact')
        return torch.cat((identity, fact)), {
            'identity_length': identity_length,
            'fact_length': fact_length,
        }

    def _metadata(self, metadata):
        if not isinstance(metadata, dict):
            raise ValueError('Lane boundary metadata is required')
        values = {}
        for name, capacity in (('identity_length', self.identity_slots),
                               ('fact_length', self.fact_slots)):
            value = metadata.get(name)
            if type(value) is not int or not 1 <= value <= capacity:
                raise ValueError(f'Invalid {name}')
            values[name] = value
        return values

    @torch.no_grad()
    def decode(self, code, metadata):
        """Decode both lanes using persisted boundary metadata."""
        if (not isinstance(code, torch.Tensor) or code.ndim != 1
                or code.numel() != self.memory_width or not code.is_floating_point()
                or not bool(torch.isfinite(code).all())):
            raise ValueError('Code has the wrong finite floating shape')
        bounds = self._metadata(metadata)
        vector = code.to(self.identity_codebook.device)
        results = {}
        start = 0
        for name, slots, slot_width, codebook, length in (
                ('identity', self.identity_slots, self.identity_slot_width,
                 self.identity_codebook, bounds['identity_length']),
                ('fact', self.fact_slots, self.fact_slot_width,
                 self.fact_codebook, bounds['fact_length'])):
            expected_slots = length + (1 if self.eos_token_id is not None else 0)
            tokens, margins = [], []
            lane_start = start
            for slot in range(expected_slots):
                lane_vector = vector[lane_start + slot * slot_width:
                                     lane_start + (slot + 1) * slot_width]
                scores = codebook @ F.normalize(lane_vector, dim=0)
                top = scores.topk(2)
                tokens.append(int(top.indices[0]))
                margins.append(float(top.values[0] - top.values[1]))
            if self.eos_token_id is not None and tokens[-1] != self.eos_token_id:
                raise ValueError(f'{name} lane did not terminate at EOS')
            results[name] = {'token_ids': tokens[:-1] if self.eos_token_id is not None else tokens,
                             'margins': margins}
            start += slots * slot_width
        return {**results, 'metadata': bounds}

    def config_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'embedding_width': self.embedding_width,
            'memory_width': self.memory_width,
            'identity_slots': self.identity_slots,
            'fact_slots': self.fact_slots,
            'total_slots': self.total_slots,
            'slot_width': self.slot_width,
            'identity_slot_width': self.identity_slot_width,
            'fact_slot_width': self.fact_slot_width,
            'seed': self.seed,
            'eos_token_id': self.eos_token_id,
        }

    @classmethod
    def from_config(cls, embedding, config):
        """Rebuild a codec from its persisted, versioned configuration."""
        if not isinstance(config, dict):
            raise ValueError('Structured text codec configuration must be a mapping')
        required = ('memory_width', 'identity_slots', 'fact_slots',
                    'identity_slot_width', 'fact_slot_width')
        if any(name not in config for name in required):
            raise ValueError('Structured text codec configuration is incomplete')
        return cls(
            embedding,
            memory_width=config['memory_width'],
            identity_slots=config['identity_slots'],
            fact_slots=config['fact_slots'],
            identity_slot_width=config['identity_slot_width'],
            fact_slot_width=config['fact_slot_width'],
            seed=config.get('seed', 29071),
            eos_token_id=config.get('eos_token_id'),
        )


class StructuredTextWriteGate:
    """Score novelty and optional query relevance before a text-lane write."""

    def __init__(self, embedding, minimum_strength=0.35):
        if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
                or not embedding.is_floating_point() or min(embedding.shape) < 2):
            raise ValueError('A floating vocabulary embedding matrix is required')
        if (isinstance(minimum_strength, bool) or not isinstance(minimum_strength, (int, float))
                or not math.isfinite(minimum_strength) or not 0 <= minimum_strength <= 1):
            raise ValueError('Minimum write strength must be finite and in [0, 1]')
        self.embedding = embedding.detach().float()
        self.minimum_strength = float(minimum_strength)

    def _mean_embedding(self, token_ids):
        if (not isinstance(token_ids, (list, tuple)) or not token_ids
                or any(type(token) is not int or not 0 <= token < self.embedding.shape[0]
                       for token in token_ids)):
            raise ValueError('Token ids must be a nonempty vocabulary sequence')
        vector = self.embedding.index_select(0, torch.tensor(token_ids, dtype=torch.long,
                                                             device=self.embedding.device)).mean(0)
        return F.normalize(vector, dim=0)

    def evaluate(self, candidate_packet, previous_packet, observation_ids, query_ids=None):
        if (not isinstance(candidate_packet, torch.Tensor) or candidate_packet.ndim != 1
                or candidate_packet.numel() < 2 or not candidate_packet.is_floating_point()
                or not bool(torch.isfinite(candidate_packet).all())):
            raise ValueError('Candidate packet must be finite and one-dimensional')
        if previous_packet is not None:
            if (not isinstance(previous_packet, torch.Tensor)
                    or previous_packet.shape != candidate_packet.shape
                    or not previous_packet.is_floating_point()
                    or not bool(torch.isfinite(previous_packet).all())):
                raise ValueError('Previous packet must match the candidate packet')
            novelty = float(1 - F.cosine_similarity(
                candidate_packet.float().unsqueeze(0), previous_packet.float().unsqueeze(0)).item())
        else:
            novelty = 1.0
        novelty = max(0.0, min(1.0, novelty))
        if query_ids is None:
            relevance = 1.0
        else:
            relevance = float((1 + F.cosine_similarity(
                self._mean_embedding(observation_ids).unsqueeze(0),
                self._mean_embedding(query_ids).unsqueeze(0)).item()) / 2)
            relevance = max(0.0, min(1.0, relevance))
        strength = max(0.0, min(1.0, novelty * relevance))
        return {
            'novelty': novelty,
            'query_relevance': relevance,
            'write_strength': strength,
            'write': strength >= self.minimum_strength,
            'minimum_strength': self.minimum_strength,
        }
