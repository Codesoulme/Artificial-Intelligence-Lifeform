"""Research-only attention over continuous slots read from native fast weights.

This module performs no vocabulary lookup and injects no answer text. The
caller still supplies addresses and the slot layout, and must independently
evaluate identity, routing, empty-memory behavior and factual generation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ContinuousMemoryAttention(nn.Module):
    def __init__(self, base, memory_context, memory_width, codec, embedding, slots, width=64,
                 condition_offset=0, output_gain=1.0):
        super().__init__()
        if (not isinstance(base, nn.Linear) or type(slots) is not int or type(width) is not int
                or type(condition_offset) is not int or width <= 0 or condition_offset < 0
                or memory_width != codec.memory_width or not 1 <= slots <= codec.slot_count
                or slots + condition_offset > codec.slot_count or isinstance(output_gain, bool)
                or not isinstance(output_gain, (int, float)) or not math.isfinite(output_gain)
                or output_gain <= 0):
            raise ValueError('Invalid continuous memory attention configuration')
        self.base = base
        self.base.requires_grad_(False)
        object.__setattr__(self, 'memory_context', memory_context)
        self.enabled = True
        self.memory_width, self.slot_width = memory_width, codec.slot_width
        self.slots, self.width, self.condition_offset = slots, width, condition_offset
        self.output_gain = float(output_gain)
        device = base.weight.device
        self.query = nn.Linear(base.in_features, width, bias=False, device=device, dtype=torch.float32)
        self.key = nn.Linear(self.slot_width, width, bias=False, device=device, dtype=torch.float32)
        self.value = nn.Linear(self.slot_width, width, bias=False, device=device, dtype=torch.float32)
        self.out = nn.Linear(width, base.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.zeros_(self.out.weight)
        self.gate = nn.Parameter(torch.tensor(-1., device=device))

    def memory_state(self):
        return {name: {k: v.detach().cpu().clone() for k, v in getattr(self, name).state_dict().items()}
                for name in ('query', 'key', 'value', 'out')} | {'gate': self.gate.detach().cpu().clone()}

    def load_memory_state(self, state):
        if not isinstance(state, dict) or set(state) != {'query', 'key', 'value', 'out', 'gate'}:
            raise ValueError('Invalid continuous memory attention state')
        for name in ('query', 'key', 'value', 'out'):
            getattr(self, name).load_state_dict(state[name], strict=True)
        with torch.no_grad():
            self.gate.copy_(state['gate'])

    def forward(self, x):
        output = self.base(x)
        active = self.memory_context.get()
        if not self.enabled or active is None:
            return output
        matrices, addresses, mask, page = active
        if mask is not None and (mask.numel() != 1 or not bool(mask[0])):
            return output
        if (x.ndim != 3 or x.shape[0] != 1 or addresses is None or len(addresses) != 2
                or len(matrices) != 2 or page is None or page.numel() != 1 or int(page[0]) != 0):
            raise ValueError('Continuous attention requires one current-memory address')
        reads = []
        for site in range(2):
            address, matrix = addresses[site], matrices[site][0]
            if address.shape != (1, self.memory_width) or matrix.shape != (1, self.memory_width, self.memory_width):
                raise ValueError('Invalid memory state or address')
            query = F.normalize(address.to(device=x.device, dtype=torch.float32), dim=-1)
            reads.append(torch.bmm(matrix.to(device=x.device, dtype=torch.float32), query.unsqueeze(-1)).squeeze(-1))
        start = self.condition_offset * self.slot_width
        end = (self.condition_offset + self.slots) * self.slot_width
        codes = ((reads[0] + reads[1]) * .5)[:, start:end].reshape(1, self.slots, self.slot_width)
        valid = codes.norm(dim=-1) > 1e-5
        if not bool(valid.any()):
            return output
        codes = F.normalize(codes, dim=-1)
        logits = torch.matmul(self.query(x.float()), self.key(codes).transpose(-1, -2)) / self.width ** .5
        weights = logits.masked_fill(~valid[:, None, :], float('-inf')).softmax(dim=-1)
        residual = self.out(torch.matmul(weights, self.value(codes))) * torch.sigmoid(self.gate) * self.output_gain
        return output + residual.to(output.dtype)
