"""Learn field access and missing-value actions for the whole-field decoder."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .research_dialogue_working_state import FIELDS, WorkingTurnState
from .research_span_action import ActionTarget


IGNORE, REQUIRE, OPTIONAL, EXCLUDE = range(4)


@dataclass(frozen=True)
class FieldAccessPlan:
    roles: tuple[int, ...]

    def __post_init__(self):
        if len(self.roles) != len(FIELDS) or any(not 0 <= role <= EXCLUDE for role in self.roles):
            raise ValueError('Expected exactly one access decision per working-state field')

    @property
    def requested(self):
        return tuple(i for i, role in enumerate(self.roles) if role in (REQUIRE, OPTIONAL))

    def project(self, state):
        """Drop fields before building keys for the generative decoder."""
        return WorkingTurnState(tuple(value if role in (REQUIRE, OPTIONAL) else ()
                                      for role, value in zip(self.roles, state.slots, strict=True)))

    def missing(self, state):
        return tuple(i for i, (role, value) in enumerate(zip(self.roles, state.slots, strict=True))
                     if role == REQUIRE and not value)


class FieldAccessHead(nn.Module):
    def __init__(self, hidden, width=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(width, len(FIELDS) * 4))

    def forward(self, question):
        return self.net(question.float()).unflatten(-1, (len(FIELDS), 4))


class MissingFieldActions(nn.Module):
    """Extend the original action distribution with one unavailable marker per field."""

    def __init__(self, hidden, width=64):
        super().__init__()
        self.hidden = nn.Linear(hidden, width, bias=False)
        self.state = nn.Linear(2 * len(FIELDS), width)
        self.residual = nn.Linear(width, hidden, bias=False)
        self.address = nn.Linear(width, len(FIELDS))
        self.mode = nn.Linear(width, 1)
        nn.init.zeros_(self.residual.weight)

    def forward(self, hidden, keys, available, unavailable, action_head, vocabulary):
        flags = torch.cat((available, unavailable), dim=-1).float()
        features = F.gelu(self.hidden(hidden.float()) + self.state(flags))
        adapted = hidden + self.residual(features).to(hidden.dtype)
        normal = action_head(adapted, keys, available, vocabulary)
        marker = self.address(features)
        marker = marker.masked_fill(~unavailable, torch.finfo(marker.dtype).min)
        gate = self.mode(features)
        gate = gate.masked_fill(~unavailable.any(dim=-1, keepdim=True), torch.finfo(gate.dtype).min)
        mark = (marker.log_softmax(dim=-1) + F.logsigmoid(gate)).masked_fill(~unavailable, -torch.inf)
        return torch.cat((normal + F.logsigmoid(-gate), mark), dim=-1)


def partial_targets(segments, vocabulary_size, eos):
    """Compile actions whose surface tokens will all be consumed by the backbone."""
    text, actions = [], []
    for address, encoded in segments:
        if address is None:
            for token in encoded:
                actions.append(ActionTarget(len(text), token))
                text.append(token)
        else:
            if not encoded or not 0 <= address < 2 * len(FIELDS):
                raise ValueError('An action requires a valid address and nonempty expression')
            actions.append(ActionTarget(len(text), vocabulary_size + address))
            text.extend(encoded)
    actions.append(ActionTarget(len(text), eos))
    return text, actions


def partial_masks(state, missing, targets, vocabulary_size, device):
    """Expose an address until its read or unavailable action has been emitted."""
    valid = [len(targets) if value else -1 for value in state.slots]
    valid.extend(len(targets) if field in missing else -1 for field in range(len(FIELDS)))
    for step, target in enumerate(targets):
        if target.action >= vocabulary_size:
            index = target.action - vocabulary_size
            if valid[index] != len(targets):
                raise ValueError('Training action references an unavailable or consumed address')
            valid[index] = step
    use = torch.arange(len(targets), device=device)[:, None] <= torch.tensor(valid, device=device)
    return use.split(len(FIELDS), dim=-1)

class TokenFieldAccessHead(nn.Module):
    """Give each field a distinct view of the question's token sequence."""

    def __init__(self, hidden, width=64):
        super().__init__()
        self.tokens = nn.Linear(hidden, width, bias=False)
        self.field_queries = nn.Parameter(torch.randn(len(FIELDS), width) / width ** .5)
        self.roles = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Dropout(.1), nn.Linear(width, 4))

    def forward(self, tokens, present):
        contextual = self.tokens(tokens.float())
        attend = contextual @ self.field_queries.T * contextual.shape[-1] ** -.5
        attend = attend.masked_fill(~present[..., None], torch.finfo(attend.dtype).min).softmax(dim=-2)
        local = torch.einsum('btf,btw->bfw', attend, contextual)
        pooled = ((contextual * present[..., None]).sum(dim=1)
                  / present.sum(dim=1, keepdim=True).clamp_min(1))
        fields = self.field_queries.expand(local.shape[0], -1, -1)
        return self.roles(torch.cat((local, pooled[:, None, :].expand_as(local), fields), dim=-1))


def typed_field_surface(encode, label, value=()):
    """Place a value or its absence under the same semantic label within one action."""
    if not label.strip():
        raise ValueError('A typed field action needs a spoken label')
    if not value:
        return tuple(encode(f'{label}尚未记录。'))
    return (*encode(f'{label}为'), *value, *encode('。'))

KEEP_SCOPE, FIRST_ONLY, SECOND_ONLY, FIRST_REQUIRED, SECOND_REQUIRED, SKIP_MEMORY = range(6)


class PairedFieldScopeHead(nn.Module):
    """Predict a joint policy whose field assignments follow the described field order."""

    def __init__(self, hidden, width=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Linear(width, 6))
        self.register_buffer('reverse', torch.tensor(
            [KEEP_SCOPE, SECOND_ONLY, FIRST_ONLY, SECOND_REQUIRED, FIRST_REQUIRED, SKIP_MEMORY]),
            persistent=False)

    def forward(self, order, swapped_order):
        direct = self.net(order.float())
        reverse = self.net(swapped_order.float()).index_select(-1, self.reverse)
        return (direct + reverse) * .5
