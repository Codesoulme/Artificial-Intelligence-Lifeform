"""Optional neural decoder for structured text memory packets.

The decoder is a research component. It reconstructs a fact sentence from a
persisted identity/fact packet without adding the packet to the user prompt.
It is not a truth verifier or a general conversational decoder.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class _TokenCodeBridge(nn.Module):
    def __init__(self, slot_width, embedding_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(slot_width, 512), nn.GELU(),
            nn.Linear(512, embedding_width))

    def forward(self, codes):
        return F.normalize(self.net(codes.float()), dim=-1)


class StructuredMemoryTextDecoder(nn.Module):
    """Generate a short fact sentence from an internal structured packet."""

    def __init__(self, embedding, *, memory_width, slot_count=5, read_slots=4, hidden=384,
                 lane_specs=None):
        super().__init__()
        if (not isinstance(embedding, torch.Tensor) or embedding.ndim != 2
                or not embedding.is_floating_point() or min(embedding.shape) < 2):
            raise ValueError('A floating vocabulary embedding matrix is required')
        if (type(memory_width) is not int or type(slot_count) is not int
                or type(read_slots) is not int or type(hidden) is not int
                or memory_width <= 0 or slot_count <= 0 or not 1 <= read_slots <= slot_count
                or hidden <= 0):
            raise ValueError('Invalid structured decoder dimensions')
        self.embedding = embedding.detach().float()
        self.vocab_size, self.embedding_width = embedding.shape
        self.memory_width = memory_width
        self.slot_count = slot_count
        self.read_slots = read_slots
        self.slot_width = memory_width // slot_count if memory_width % slot_count == 0 else None
        self.hidden_size = hidden
        if lane_specs is None:
            if self.slot_width is None:
                raise ValueError('Memory width must divide into equal slots without lane specs')
            self.lane_specs = None
            self.bridge = _TokenCodeBridge(self.slot_width, self.embedding_width)
            encoder_width = read_slots * self.embedding_width
        else:
            if (not isinstance(lane_specs, (tuple, list)) or not lane_specs
                    or any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in lane_specs)):
                raise ValueError('Lane specs must contain (slot_count, slot_width) pairs')
            if sum(slots * width for slots, width in lane_specs) != memory_width:
                raise ValueError('Lane specs must cover the memory width')
            self.lane_specs = tuple((int(slots), int(width)) for slots, width in lane_specs)
            if any(slots <= 0 or width < 2 for slots, width in self.lane_specs):
                raise ValueError('Lane specs must be positive')
            self.identity_bridge = _TokenCodeBridge(self.lane_specs[0][1], self.embedding_width)
            self.fact_bridge = _TokenCodeBridge(self.lane_specs[1][1], self.embedding_width)
            encoder_width = sum(slots for slots, _ in self.lane_specs) * self.embedding_width
        self.encoder_width = encoder_width
        self.memory_encoder = nn.Sequential(
            nn.Linear(encoder_width, hidden), nn.Tanh())
        self.gru = nn.GRUCell(self.embedding_width, hidden)
        self.to_embedding = nn.Linear(hidden, self.embedding_width, bias=False)

    @classmethod
    def from_checkpoint(cls, embedding, path, *, expected_source_sha256=None,
                        expected_base_sha256=None, expected_format=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if not isinstance(payload, dict) or payload.get('format') not in (
                None, 'spora-identity-codec-composition-v1',
                'spora-structured-memory-text-decoder-v1'):
            raise ValueError('Unsupported structured text decoder checkpoint')
        if expected_format is not None and payload.get('format') != expected_format:
            raise ValueError('Structured decoder format is not valid for this runtime')
        if expected_source_sha256 is not None and payload.get('source_sha256') != expected_source_sha256:
            raise ValueError('Structured decoder source checkpoint mismatch')
        if expected_base_sha256 is not None and payload.get('base_sha256') != expected_base_sha256:
            raise ValueError('Structured decoder base model mismatch')
        suffix = payload.get('suffix', payload.get('state'))
        bridge = payload.get('bridge')
        if not isinstance(suffix, dict):
            raise ValueError('Structured decoder checkpoint is incomplete')
        embedding_width = suffix['to_embedding.weight'].shape[0]
        if embedding.shape[1] != embedding_width:
            raise ValueError('Structured decoder embedding width mismatch')
        memory_encoder_width = suffix['memory_encoder.0.weight'].shape[1]
        if memory_encoder_width % embedding_width:
            raise ValueError('Structured decoder memory width is not slot-aligned')
        read_slots = memory_encoder_width // embedding_width
        if payload.get('format') == 'spora-structured-memory-text-decoder-v1':
            config = payload.get('codec', {})
            required = ('memory_width', 'identity_slots', 'fact_slots',
                        'identity_slot_width', 'fact_slot_width')
            if any(name not in config for name in required):
                raise ValueError('Structured decoder codec configuration is incomplete')
            lane_specs = ((config['identity_slots'], config['identity_slot_width']),
                          (config['fact_slots'], config['fact_slot_width']))
            decoder = cls(embedding, memory_width=config['memory_width'], slot_count=sum(config[name] for name in ('identity_slots', 'fact_slots')),
                          read_slots=read_slots,
                          hidden=suffix['memory_encoder.0.weight'].shape[0],
                          lane_specs=lane_specs)
        else:
            if not isinstance(bridge, dict):
                raise ValueError('Legacy structured decoder bridge is missing')
            slot_width = bridge['net.0.weight'].shape[1]
            decoder = cls(embedding, memory_width=slot_width * 5, slot_count=5,
                          read_slots=read_slots,
                          hidden=suffix['memory_encoder.0.weight'].shape[0])
        decoder.load_state_dict(suffix, strict=True)
        return decoder

    def decode_memory(self, memory):
        if (not isinstance(memory, torch.Tensor) or memory.ndim != 2
                or memory.shape[-1] != self.memory_width or not memory.is_floating_point()
                or not bool(torch.isfinite(memory).all())):
            raise ValueError('Memory packet has the wrong shape or contains nonfinite values')
        if self.lane_specs is None:
            slots = memory[:, :self.read_slots * self.slot_width].reshape(
                memory.shape[0], self.read_slots, self.slot_width)
            slots = F.normalize(slots, dim=-1)
            encoded = self.bridge(slots).reshape(memory.shape[0], -1)
        else:
            offset = 0
            encoded_lanes = []
            for index, (slots, width) in enumerate(self.lane_specs):
                lane = memory[:, offset:offset + slots * width].reshape(
                    memory.shape[0], slots, width)
                lane = F.normalize(lane, dim=-1)
                bridge = self.identity_bridge if index == 0 else self.fact_bridge
                encoded_lanes.append(bridge(lane).reshape(memory.shape[0], -1))
                offset += slots * width
            encoded = torch.cat(encoded_lanes, dim=-1)
        return self.memory_encoder(encoded.float())

    def forward(self, memory, target_ids):
        if (not isinstance(target_ids, torch.Tensor) or target_ids.ndim != 2
                or target_ids.dtype != torch.long or target_ids.shape[0] != memory.shape[0]):
            raise ValueError('Target ids must be a matching batch of long sequences')
        hidden = self.decode_memory(memory)
        outputs = []
        for position in range(target_ids.shape[1]):
            outputs.append(self.to_embedding(hidden) @ self.embedding.T)
            if position + 1 < target_ids.shape[1]:
                token_embedding = self.embedding.index_select(0, target_ids[:, position])
                hidden = self.gru(token_embedding.float(), hidden)
        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def generate(self, memory, *, eos_id, max_new_tokens=32):
        if type(eos_id) is not int or not 0 <= eos_id < self.vocab_size:
            raise ValueError('EOS id is outside the decoder vocabulary')
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
            raise ValueError('Invalid generation budget')
        hidden = self.decode_memory(memory)
        tokens = []
        for _ in range(max_new_tokens):
            next_id = self.to_embedding(hidden).matmul(self.embedding.T).argmax(-1)
            tokens.append(next_id)
            hidden = self.gru(self.embedding.index_select(0, next_id).float(), hidden)
            if bool((next_id == eos_id).all()):
                break
        return torch.stack(tokens, dim=1)
