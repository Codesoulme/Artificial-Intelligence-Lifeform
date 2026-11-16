"""Experimental ordered packet serialization; not an autonomous memory system.

The caller supplies identity and token boundaries. Packets and their ordering
must remain available; this module neither learns facts nor routes questions.
"""
from dataclasses import dataclass

import torch

from spora.structured_text_memory import StructuredTextMemoryCodec


@dataclass(frozen=True)
class TextChunk:
    index: int
    total: int
    packet: torch.Tensor
    metadata: dict


def encode_chunks(codec, identity_ids, fact_ids, *, max_chunks):
    """Serialize a bounded, nonempty fact into explicitly ordered packets."""
    if not isinstance(codec, StructuredTextMemoryCodec):
        raise ValueError('Expected structured text codec')
    if type(max_chunks) is not int or max_chunks < 1:
        raise ValueError('max_chunks must be a positive integer')
    # Validate before allocating or returning any partial result.
    identity = codec._ids(identity_ids, 'Identity')
    fact = codec._ids(fact_ids, 'Fact')
    capacity = codec.fact_slots - (codec.eos_token_id is not None)
    if capacity < 1:
        raise ValueError('Fact lane cannot hold content')
    count = (len(fact) + capacity - 1) // capacity
    if count > max_chunks:
        raise ValueError('Text exceeds maximum chunk count')
    chunks = []
    for index in range(count):
        packet, metadata = codec.encode(identity, fact[index * capacity:(index + 1) * capacity])
        chunks.append(TextChunk(index, count, packet.detach().clone(), dict(metadata)))
    return tuple(chunks)


def decode_chunks(codec, chunks, *, max_chunks):
    """Recover a complete ordered record; reject missing, reordered or mixed identities."""
    if not isinstance(codec, StructuredTextMemoryCodec):
        raise ValueError('Expected structured text codec')
    if type(max_chunks) is not int or max_chunks < 1:
        raise ValueError('max_chunks must be a positive integer')
    if not isinstance(chunks, (list, tuple)) or not 1 <= len(chunks) <= max_chunks:
        raise ValueError('Invalid chunk count')
    identity, fact = None, []
    for index, chunk in enumerate(chunks):
        if (not isinstance(chunk, TextChunk) or type(chunk.index) is not int
                or type(chunk.total) is not int or chunk.index != index or chunk.total != len(chunks)):
            raise ValueError('Missing, reordered or malformed chunk')
        decoded = codec.decode(chunk.packet, chunk.metadata)
        item_identity = decoded['identity']['token_ids']
        if identity is None:
            identity = item_identity
        elif item_identity != identity:
            raise ValueError('Chunk identity mismatch')
        fact.extend(decoded['fact']['token_ids'])
    return {'identity_ids': identity, 'fact_ids': fact}
